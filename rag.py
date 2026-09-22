"""Módulo RAG: carga el manual de San Cristóbal, lo fragmenta y lo indexa.

Pipeline:
    1. Cargar el PDF con PyPDFLoader.
    2. Fragmentar en chunks (con solapamiento).
    3. Etiquetar cada chunk según la SECCIÓN del manual (por página, no por
       keywords frágiles).
    4. Generar embeddings (HuggingFace) y persistir en Chroma.

La base vectorial se construye una sola vez y se reutiliza.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PDF_PATH,
    ensure_dirs,
)

# Los módulos pesados (chromadb, sentence-transformers/torch, pypdf) se
# importan de forma LAZY dentro de las funciones que los usan, para que
# importar ``agent`` no demore ~10s en arrancar.
if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Mapeo página -> sección del manual
# ---------------------------------------------------------------------------
# Los índices de página son 0-based (los que devuelve PyPDF en metadata).
# El manual organiza los productos en secciones claras:
#   - Auto base (A)        -> páginas 13-14
#   - Auto Mix (B)         -> página 15
#   - Terceros completos   -> páginas 16-18  (C, C+ y CM comparten base)
#   - Auto extra / Todo R. -> páginas 19-20  (D)
#   - El resto (índice, franquicia, exclusiones, asistencia) -> General
def pagina_a_seccion(pagina: int) -> str:
    """Devuelve la sección del manual para un índice de página (0-based)."""
    if 12 <= pagina <= 13:
        return "A"  # Auto base (Responsabilidad Civil)
    if pagina == 14:
        return "B"  # Auto Mix (Daños totales)
    if 15 <= pagina <= 17:
        return "C"  # Terceros completos (C / C+ / CM)
    if 18 <= pagina <= 19:
        return "D"  # Auto extra / Todo Riesgo
    return "General"


# ---------------------------------------------------------------------------
# Construcción / carga de la base vectorial
# ---------------------------------------------------------------------------
_vector_store: Chroma | None = None
_chunks_cache: list[Document] | None = None


class E5Embeddings:
    """Embeddings e5 con prefijos query/passage (carga lazy del modelo).

    No hereda de ``HuggingFaceEmbeddings`` para no importar
    sentence-transformers/torch al importar este módulo. Expone la misma
    interfaz (``embed_documents`` / ``embed_query``) que espera Chroma.
    """

    def __init__(self, model_name: str, **kwargs):
        from langchain_huggingface import HuggingFaceEmbeddings  # lazy

        self._inner = HuggingFaceEmbeddings(model_name=model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"query: {text}")


def _embeddings() -> E5Embeddings:
    # Normalizamos los vectores para que la similitud coseno sea correcta.
    return E5Embeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def construir_vector_store(forzar: bool = False) -> Chroma:
    """Construye (o carga) la base vectorial de Chroma."""
    # Lazy imports: así importar ``agent`` no carga torch/chromadb/pypdf.
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    ensure_dirs()

    if not forzar and CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"Cargando base vectorial existente desde {CHROMA_DIR}")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=_embeddings(),
        )

    print(f"Cargando PDF desde {PDF_PATH}")
    loader = PyPDFLoader(str(PDF_PATH))
    documentos = loader.load()
    # Excluimos el índice (páginas 0-2, 0-based): concentra todas las
    # keywords ("granizo", "robo", "incendio") en poco texto y contamina
    # la búsqueda semántica. No aporta cláusulas.
    documentos = [d for d in documentos if d.metadata.get("page", 0) >= 3]
    print(f"PDF cargado: {len(documentos)} páginas (sin índice).")

    # Chunking ESTRUCTURAL: dividimos por los ítems numerados del manual
    # ("N)" al inicio de línea) para que títulos como "10) Granizo" queden
    # junto a su contenido. Un splitter por caracteres cortaba justo entre
    # el título y el cuerpo, degradando la recuperación semántica.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Priorizamos saltos lógicos: párrafo, oración, línea, palabra.
        separators=["\n\n", ". ", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = []
    for doc in documentos:
        partes = re.split(r"(?=^\d+\))", doc.page_content, flags=re.MULTILINE)
        for parte in partes:
            parte = parte.strip()
            if not parte:
                continue
            if len(parte) > CHUNK_SIZE:
                # Ítems muy largos: los sub-dividimos por caracteres.
                sub = splitter.split_documents(
                    [Document(page_content=parte, metadata=dict(doc.metadata))]
                )
                chunks.extend(sub)
            else:
                chunks.append(Document(page_content=parte, metadata=dict(doc.metadata)))
    print(f"Documento fragmentado en {len(chunks)} chunks.")

    # Etiquetamos cada chunk con su sección según la página.
    for chunk in chunks:
        pagina = chunk.metadata.get("page", 0)
        chunk.metadata["seccion"] = pagina_a_seccion(pagina)

    # Cacheamos los chunks para poder armar el BM25 (retrieval híbrido).
    global _chunks_cache
    _chunks_cache = chunks

    store = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Base vectorial guardada en {CHROMA_DIR}")
    return store


def get_vector_store(forzar: bool = False) -> Chroma:
    """Singleton perezoso de la base vectorial."""
    global _vector_store
    if _vector_store is None or forzar:
        _vector_store = construir_vector_store(forzar)
    return _vector_store


# ---------------------------------------------------------------------------
# Recuperación (retrieval)
# ---------------------------------------------------------------------------
def _tokenizar(texto: str) -> list[str]:
    """Tokeniza en minúsculas, extrayendo solo palabras y números.

    El tokenizer por defecto de BM25 (``text.split()``) es sensible a
    mayúsculas y no quita puntuación: "granizo" no matcheaba "Granizo" ni
    "granizo?". Este lo normaliza.
    """
    return re.findall(r"[a-záéíóúüñ0-9]+", texto.lower())


def _obtener_chunks() -> list[Document]:
    """Devuelve los chunks indexados (los reconstruye desde Chroma si hace falta)."""
    from langchain_core.documents import Document  # lazy

    global _chunks_cache
    if _chunks_cache is not None:
        return _chunks_cache
    store = get_vector_store()
    res = store._collection.get(include=["documents", "metadatas"])
    _chunks_cache = [
        Document(page_content=d, metadata=m)
        for d, m in zip(res["documents"], res["metadatas"])
    ]
    return _chunks_cache


def buscar_clausulas(consulta: str, k: int = 6) -> str:
    """Busca cláusulas relevantes con RETRIEVAL HÍBRIDO y las devuelve formateadas.

    Combina:
      1. BM25 (matching léxico): captura términos exactos como "granizo" que
         los embeddings semánticos a veces no discriminan bien.
      2. Búsqueda semántica (Chroma + embeddings): captura significado.

    Se fusionan con EnsembleRetriever (RRF). Busca en TODO el manual (sin
    filtro duro por producto) y devuelve cada chunk con su tag de sección.
    """
    from langchain_classic.retrievers import EnsembleRetriever  # lazy
    from langchain_community.retrievers import BM25Retriever  # lazy

    store = get_vector_store()
    chunks = _obtener_chunks()

    retriever_bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_tokenizar)
    retriever_bm25.k = k
    retriever_vector = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    ensemble = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vector],
        weights=[0.5, 0.5],
    )
    docs = ensemble.invoke(consulta)
    if not docs:
        return "No se encontraron cláusulas relevantes para esa consulta."

    partes = []
    for doc in docs:
        seccion = doc.metadata.get("seccion", "Sin sección")
        pagina = doc.metadata.get("page", "?")
        partes.append(
            f"[Sección: {seccion} | Pág: {pagina}]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(partes)
