import json
import os
import time
import logging
import mlflow
import functools
import hashlib
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks import CBEventType

try:
    from noworkflow.now.collection.prov_execution import collector
    NOWORKFLOW_AVAILABLE = True
except ImportError:
    NOWORKFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Captura o hash curto do commit atual para rastreabilidade do código-fonte."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _resolve_noworkflow_trial_id() -> Optional[str]:
    """
    Tenta obter o trial_id do noWorkflow em duas etapas:
    1. Via objeto collector em memória (disponível durante `now run`)
    2. Via variável de ambiente NOW_TRIAL_ID (fallback manual)
    """
    if not NOWORKFLOW_AVAILABLE:
        return None
    if hasattr(collector, "trial_id") and collector.trial_id:
        return str(collector.trial_id)
    return os.environ.get("NOW_TRIAL_ID")


def _resolve_noworkflow_trial_id_post() -> Optional[str]:
    """
    Fallback pós-execução: lê o trial mais recente direto do SQLite do noWorkflow.
    Usado quando o collector ainda não expôs o trial_id durante a execução.
    """
    if not NOWORKFLOW_AVAILABLE:
        return None
    if os.environ.get("RUN_MODE") != "noworkflow":
        return None
    try:
        import sqlite3
        now_db = os.path.join(".noworkflow", "db.sqlite")
        if os.path.exists(now_db):
            conn = sqlite3.connect(now_db)
            row = conn.execute("SELECT id FROM trial ORDER BY start DESC LIMIT 1").fetchone()
            conn.close()
            if row:
                return str(row[0])
    except Exception:
        pass
    return None


def _save_log_and_artifact(log: Dict[str, Any], prefix: str, run_id_short: str) -> str:
    """Salva o JSON de proveniência em disco e registra como artefato no MLflow."""
    os.makedirs("data/provenance_logs", exist_ok=True)
    filename = f"data/provenance_logs/{prefix}_{run_id_short}_{int(time.time())}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4, ensure_ascii=False, default=str)
    mlflow.log_artifact(filename)
    return filename


class LINAProvenanceHandler(BaseCallbackHandler):
    """
    Handler de Callbacks nativo do LlamaIndex adaptado para e-Science.
    Captura proveniência retrospectiva de dados (Retrieval e LLM Generation).
    """

    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.reset_trace()

    def reset_trace(self) -> None:
        self.trace_data: Dict[str, Any] = {
            "retrieval_events": [],
            "llm_events": [],
            "timestamps": {},
            "stage_timestamps": {},
            "latency_seconds": 0.0,
        }

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.QUERY:
            self.trace_data["timestamps"]["query_start"] = time.time()
            self.trace_data["user_query"] = payload.get("query_str") if payload else None
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if not payload:
            return

        if event_type == CBEventType.QUERY:
            self.trace_data["timestamps"]["query_end"] = time.time()
            ts = self.trace_data["timestamps"]
            if "query_start" in ts and "query_end" in ts:
                self.trace_data["latency_seconds"] = ts["query_end"] - ts["query_start"]

        elif event_type == CBEventType.RETRIEVE:
            self.trace_data["stage_timestamps"]["retrieval_end"] = time.time()
            nodes = payload.get("nodes", [])
            self.trace_data["retrieval_events"].append({
                "event_id": event_id,
                "timestamp": datetime.now().isoformat(),
                "nodes_count": len(nodes),
                "nodes": [
                    {
                        "node_id": node.node.node_id,
                        "score": node.score if hasattr(node, "score") else None,
                        "file_name": node.node.metadata.get("file_name"),
                        "source": node.node.metadata.get("source"),
                        "text_snippet": node.node.text[:200] if node.node.text else None,
                    }
                    for node in nodes
                ],
            })

        elif event_type == CBEventType.LLM:
            self.trace_data["stage_timestamps"]["llm_end"] = time.time()
            response = payload.get("response")
            prompt_messages = payload.get("messages", [])
            usage = self._extract_token_usage(response)

            self.trace_data["llm_events"].append({
                "event_id": event_id,
                "timestamp": datetime.now().isoformat(),
                "output_text": str(response),
                "token_usage": usage,
                "prompt_preview": str(prompt_messages)[:500] if prompt_messages else None,
            })

    def _extract_token_usage(self, response) -> Dict[str, int]:
        if response is None:
            return {}
        if hasattr(response, "additional_kwargs"):
            token_counts = response.additional_kwargs.get("token_counts")
            if token_counts:
                return token_counts
        if hasattr(response, "raw") and hasattr(response.raw, "usage"):
            raw_usage = response.raw.usage
            if hasattr(raw_usage, "model_dump"):
                return raw_usage.model_dump()
            if hasattr(raw_usage, "__dict__"):
                return dict(raw_usage.__dict__)
        return {}

    def start_trace(self, trace_id: Optional[str] = None, **kwargs: Any) -> None:
        self.reset_trace()

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def get_provenance_data(self) -> Dict[str, Any]:
        return self.trace_data


def track_provenance_experiment(experiment_name: str, prospective_params: Dict[str, Any]):
    """
    Decorador para o pipeline de consulta (query engine).
    Captura proveniência prospectiva (parâmetros) e retrospectiva (eventos LlamaIndex)
    e registra tudo no MLflow, vinculando ao trial do noWorkflow quando disponível.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from llama_index.core import Settings
            handler = LINAProvenanceHandler()
            Settings.callback_manager.add_handler(handler)

            mlflow.set_experiment(experiment_name)
            noworkflow_trial_id = _resolve_noworkflow_trial_id()

            with mlflow.start_run() as ml_run:
                if prospective_params:
                    mlflow.log_params(prospective_params)

                mlflow.set_tag("framework", "llama_index")
                mlflow.set_tag("function_name", func.__name__)
                mlflow.set_tag("experiment_name", experiment_name)

                query_str = args[0] if args else ""
                query_hash = hashlib.md5(str(query_str).encode()).hexdigest()[:8]
                mlflow.set_tag("user_query_hash", query_hash)
                mlflow.set_tag("user_query_short", str(query_str)[:50])

                if noworkflow_trial_id is not None:
                    mlflow.set_tag("noworkflow_trial_id", str(noworkflow_trial_id))

                start_clock = time.perf_counter()
                execution_error = None
                result = None
                try:
                    result = func(*args, **kwargs)
                    mlflow.set_tag("execution_status", "success")
                except Exception as e:
                    execution_error = e
                    mlflow.set_tag("execution_status", "error")
                    raise
                finally:
                    wall_time = time.perf_counter() - start_clock
                    retrospect_data = handler.get_provenance_data()

                    latency = retrospect_data.get("latency_seconds", 0.0)
                    mlflow.log_metric("latency_seconds", latency)
                    mlflow.log_metric("wall_time_seconds", wall_time)

                    st = retrospect_data.get("stage_timestamps", {})
                    q_start = retrospect_data["timestamps"].get("query_start", 0)
                    retrieval_end = st.get("retrieval_end", 0)
                    llm_end = st.get("llm_end", 0)

                    retrieval_latency = retrieval_end - q_start if retrieval_end and q_start else 0
                    llm_latency = llm_end - retrieval_end if llm_end and retrieval_end else 0

                    mlflow.log_metric("retrieval_latency_seconds", retrieval_latency)
                    mlflow.log_metric("llm_latency_seconds", llm_latency)
                    mlflow.log_metric(
                        "nodes_retrieved",
                        sum(e["nodes_count"] for e in retrospect_data["retrieval_events"]),
                    )

                    if retrospect_data["llm_events"]:
                        usage = retrospect_data["llm_events"][0].get("token_usage", {})
                        mlflow.log_metric("total_tokens", usage.get("total_tokens", 0))

                    all_scores = [
                        n["score"]
                        for ev in retrospect_data["retrieval_events"]
                        for n in ev["nodes"]
                        if n.get("score")
                    ]
                    if all_scores:
                        mlflow.log_metric("avg_retrieval_score", sum(all_scores) / len(all_scores))

                    # Fallback pós-execução para o trial_id
                    if noworkflow_trial_id is None:
                        noworkflow_trial_id_post = _resolve_noworkflow_trial_id_post()
                        if noworkflow_trial_id_post:
                            mlflow.set_tag("noworkflow_trial_id", noworkflow_trial_id_post)
                    else:
                        noworkflow_trial_id_post = noworkflow_trial_id

                    run_log = {
                        "experiment": experiment_name,
                        "mlflow_run_id": ml_run.info.run_id,
                        "execution_timestamp": datetime.now().isoformat(),
                        "noworkflow_trial_id": noworkflow_trial_id_post or noworkflow_trial_id,
                        "prospect_parameters": prospective_params,
                        "retrospect_execution": {
                            **retrospect_data,
                            "wall_time_seconds": wall_time,
                            "overhead_provenance_seconds": wall_time - latency,
                            "retrieval_latency_seconds": retrieval_latency,
                            "llm_latency_seconds": llm_latency,
                        },
                        "error": str(execution_error) if execution_error else None,
                    }

                    file_id = (noworkflow_trial_id_post or noworkflow_trial_id or ml_run.info.run_id[:8])
                    _save_log_and_artifact(run_log, prefix="query", run_id_short=file_id)

                    try:
                        Settings.callback_manager.remove_handler(handler)
                    except Exception:
                        pass

                return result
        return wrapper
    return decorator


def _hash_file(filepath: str) -> str:
    """Calcula o MD5 de um arquivo para garantir rastreabilidade do conteúdo."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_document_fingerprints(documents) -> List[Dict[str, Any]]:
    """
    Extrai metadados únicos por arquivo fonte a partir dos documentos carregados.
    Cada arquivo aparece uma única vez, com seu hash MD5 e contagem de páginas.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for doc in documents:
        filepath = doc.metadata.get("file_name", "")
        if filepath and filepath not in seen:
            entry: Dict[str, Any] = {"file_path": filepath}
            if os.path.exists(filepath):
                entry["md5"] = _hash_file(filepath)
                entry["size_bytes"] = os.path.getsize(filepath)
            else:
                entry["md5"] = "file_not_found"
                entry["size_bytes"] = 0
            entry["pages_loaded"] = 0
            seen[filepath] = entry
        if filepath in seen:
            seen[filepath]["pages_loaded"] += 1
    return list(seen.values())


def track_ingestion_experiment(experiment_name: str, prospective_params: Dict[str, Any]):
    """
    Decorador para o pipeline de ingestão de documentos.

    Captura:
    - Prospectivo: parâmetros da pipeline (window_size, embedding, coleção, git commit)
    - Retrospectivo: documentos ingeridos com MD5, nodes gerados, wall_time

    O mlflow_run_id gerado aqui deve ser usado como `index_version` nos
    prospective_params do query engine para fechar o grafo de linhagem:
        documento PDF → ingestão → índice Qdrant → consulta → resposta
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            noworkflow_trial_id = _resolve_noworkflow_trial_id()

            # O primeiro argumento de run_pipeline é sempre a lista de documentos
            documents = args[0] if args else []

            with mlflow.start_run() as ml_run:
                if prospective_params:
                    mlflow.log_params(prospective_params)

                mlflow.set_tag("framework", "llama_index")
                mlflow.set_tag("function_name", func.__name__)
                mlflow.set_tag("experiment_name", experiment_name)
                mlflow.log_metric("documents_pages_loaded", len(documents))

                if noworkflow_trial_id is not None:
                    mlflow.set_tag("noworkflow_trial_id", str(noworkflow_trial_id))

                # Coleta fingerprints dos arquivos ANTES de rodar a pipeline
                doc_fingerprints = _collect_document_fingerprints(documents)

                start_clock = time.perf_counter()
                execution_error = None
                nodes = None
                try:
                    nodes = func(*args, **kwargs)
                    mlflow.set_tag("execution_status", "success")
                except Exception as e:
                    execution_error = e
                    mlflow.set_tag("execution_status", "error")
                    raise
                finally:
                    wall_time = time.perf_counter() - start_clock
                    nodes_count = len(nodes) if nodes is not None else 0

                    mlflow.log_metric("wall_time_seconds", wall_time)
                    mlflow.log_metric("nodes_generated", nodes_count)
                    mlflow.log_metric("source_files_count", len(doc_fingerprints))

                    # Fallback pós-execução para o trial_id
                    if noworkflow_trial_id is None:
                        noworkflow_trial_id_post = _resolve_noworkflow_trial_id_post()
                        if noworkflow_trial_id_post:
                            mlflow.set_tag("noworkflow_trial_id", noworkflow_trial_id_post)
                    else:
                        noworkflow_trial_id_post = noworkflow_trial_id

                    run_log = {
                        "experiment": experiment_name,
                        "mlflow_run_id": ml_run.info.run_id,
                        "execution_timestamp": datetime.now().isoformat(),
                        "noworkflow_trial_id": noworkflow_trial_id_post or noworkflow_trial_id,
                        "prospect_parameters": prospective_params,
                        "retrospect_execution": {
                            "wall_time_seconds": wall_time,
                            "documents_pages_loaded": len(documents),
                            "source_files_count": len(doc_fingerprints),
                            "nodes_generated": nodes_count,
                            "source_files": doc_fingerprints,
                        },
                        "error": str(execution_error) if execution_error else None,
                    }

                    file_id = (noworkflow_trial_id_post or noworkflow_trial_id or ml_run.info.run_id[:8])
                    _save_log_and_artifact(run_log, prefix="ingest", run_id_short=file_id)

                    logger.info(
                        f"[Provenance] Ingestão concluída — "
                        f"{nodes_count} nodes | {len(doc_fingerprints)} arquivo(s) | "
                        f"{wall_time:.2f}s | mlflow_run_id={ml_run.info.run_id}"
                    )

                return nodes
        return wrapper
    return decorator


def track_execution(func):
    """Decorador legado — substituído por track_ingestion_experiment."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[Provenance] Executing: {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"[Provenance] Completed: {func.__name__}")
        return result
    return wrapper