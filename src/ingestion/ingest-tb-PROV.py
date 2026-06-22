import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Union
import httpx

from core.config import RAGConfig
from core.embeddings import Embedding
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import BaseNode, Document
from core.util import QdrantUtil

from provenance.tracker import track_ingestion_experiment, get_git_commit

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

config = RAGConfig


def get_documents(
    input_dir: Optional[Union[Path, str]] = None,
) -> List[Document]:
    """Carrega documentos PDF do diretório de entrada com metadados de fonte."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory '{input_dir}' doesn't exist")

    logger.info(f"Load documents from '{input_dir}'")

    def set_metadata(filename):
        return {"file_name": filename, "source": "user_pdfs"}

    parser = PyMuPDFReader()
    file_extractor = {".pdf": parser}

    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        required_exts=[".pdf"],
        file_metadata=set_metadata,
        file_extractor=file_extractor,
    ).load_data()

    logger.info(f"Found {len(documents)} page(s)")
    return documents


@track_ingestion_experiment(
    experiment_name="Ingestao_RAG",
    prospective_params={
        "window_size": 3,
        "window_metadata_key": "ContextWindow",
        "include_prev_next_rel": True,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "qdrant_collection": config.QDRANT_COLLECTION_TB,
        "docstore_strategy": "UPSERTS",
        "git_commit": get_git_commit(),
    },
)
def run_pipeline(documents: List[Document]) -> Sequence[BaseNode]:
    qdrant_client = QdrantUtil.get_client(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=config.REQUEST_TIMEOUT,
    )

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        include_prev_next_rel=True,
        window_metadata_key="ContextWindow",
        original_text_metadata_key="node_text",
    )

    pipeline = IngestionPipeline(
        transformations=[
            node_parser,
            # KeywordExtractor(llm, show_progress=False),  # Pausado para otimização
            Embedding(config).get_embedding_model(),
        ],
        docstore_strategy=DocstoreStrategy.UPSERTS,
        vector_store=QdrantUtil.get_vectorstore(
            client=qdrant_client,
            collection_name=config.QDRANT_COLLECTION_TB,
        ),
    )
    nodes = pipeline.run(documents=documents)
    return nodes


def main():
    logger.info("Starting ingestion process")
    logger.info("Using SentenceWindowNodeParser (window_size=3)")
    logger.info(f"Using embedding model '{config.EMBEDDING_MODEL}'")
    logger.info(f"Git commit: {get_git_commit()}")

    try:
        response = httpx.get(
            f"{config.QDRANT_URL}/collections",
            headers={"api-key": config.QDRANT_API_KEY},
        )
        response.raise_for_status()
        logger.info(f"Qdrant status: {response.status_code} - {response.json()}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Erro HTTP ao conectar ao Qdrant: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Erro inesperado ao conectar ao Qdrant: {e}")

    documents = get_documents(input_dir=os.path.join(os.getcwd(), config.TB_DOCS_PATH))
    nodes = run_pipeline(documents=documents)

    logger.info(f"Ingested {len(nodes)} node(s)")
    logger.info("Ingestion process completed")


if __name__ == "__main__":
    main()