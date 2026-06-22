import logging
import sys
import subprocess
import os

from core.config import RAGConfig
from dotenv import load_dotenv
from provenance.tracker import track_provenance_experiment
from core.embeddings import Embedding
from core.llm import LLM
from core.util import QdrantUtil

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    style="%",
    level=logging.INFO,
)

load_dotenv()
config = RAGConfig

# Inicializa modelos globalmente para evitar re-inicialização a cada chamada
Settings.embed_model = Embedding(config).get_embedding_model()
Settings.llm = LLM(config).get_llm()


def get_git_commit() -> str:
    """Captura o hash curto do commit atual para rastreabilidade do código-fonte."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# Lê a variável de ambiente para distinguir execuções Python puro vs noWorkflow
runner_mode = os.environ.get("RUN_MODE", "python")


@track_provenance_experiment(
    experiment_name="Validacao_RAG_CLI",
    prospective_params={
        "similarity_top_k": 5,
        "similarity_cutoff": 0.20,
        "qdrant_collection": config.QDRANT_COLLECTION_TB,
        "llm_model": config.OPEN_API_MODEL,
        "llm_temperature": 0.0,
        "embedding_model": "text-embedding-ada-002",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "git_commit": get_git_commit(),
        "runner": runner_mode,
    },
)
def query(query_text: str) -> RESPONSE_TYPE:
    qdrant_client = QdrantUtil.get_client(
        url=config.QDRANT_URL,
        timeout=config.REQUEST_TIMEOUT,
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION_TB,
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,
    )

    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.20)

    metadata_replacement_postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="ContextWindow"
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor, metadata_replacement_postprocessor],
        # Passa o callback_manager explicitamente para garantir captura dos eventos
        callback_manager=Settings.callback_manager,
    )

    return query_engine.query(query_text)


def main():
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
        result = query(query_text)
        pprint_response(result, show_source=True)


if __name__ == "__main__":
    main()