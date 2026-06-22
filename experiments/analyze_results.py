import mlflow
import pandas as pd
import numpy as np

# Configura o acesso ao banco local do MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def remove_outliers(values):
    """Remove o maior e o menor valor, conforme metodologia do plano (Item 2.2)."""
    s = sorted(values)
    return s[1:-1] if len(s) >= 3 else s

def analyze():
    print("="*60)
    print("RESULTADOS DO EXPERIMENTO (OVERHEAD DE PROVENIÊNCIA)")
    print("="*60)
    
    # Busca todas as execuções do experimento
    runs = mlflow.search_runs(experiment_names=["Validacao_RAG_CLI"])
    
    if runs.empty:
        print("Nenhuma execução encontrada. Rode o run_battery.sh primeiro!")
        return

    # Filtra por grupos
    python_runs = runs[runs["params.runner"] == "python"]
    now_runs = runs[runs["params.runner"] == "noworkflow"]

    print(f"Execuções Python puro encontradas: {len(python_runs)}")
    print(f"Execuções noWorkflow encontradas: {len(now_runs)}")
    print("-" * 60)

    # Identificar todas as perguntas distintas que foram rodadas
    queries = runs["tags.user_query_short"].dropna().unique()

    total_py_mean = []
    total_nw_mean = []

    for query in queries:
        py = python_runs[python_runs["tags.user_query_short"] == query]
        nw = now_runs[now_runs["tags.user_query_short"] == query]
        
        # Coleta e remove outliers do Wall Time
        py_times = remove_outliers(py["metrics.wall_time_seconds"].dropna().tolist())
        nw_times = remove_outliers(nw["metrics.wall_time_seconds"].dropna().tolist())
        
        if not py_times or not nw_times:
            continue
            
        py_mean = np.mean(py_times)
        nw_mean = np.mean(nw_times)
        overhead_pct = ((nw_mean - py_mean) / py_mean) * 100
        
        total_py_mean.append(py_mean)
        total_nw_mean.append(nw_mean)
        
        # Pega a latência média por etapa para o noWorkflow (Para gerar a Figura 2 do plano)
        nw_retrieval = np.mean(remove_outliers(nw["metrics.retrieval_latency_seconds"].dropna().tolist()))
        nw_llm = np.mean(remove_outliers(nw["metrics.llm_latency_seconds"].dropna().tolist()))
        
        print(f"Pergunta: '{query}'")
        print(f"   ► Python Puro: {py_mean:.3f}s (± {np.std(py_times):.3f}s)")
        print(f"   ► noWorkflow : {nw_mean:.3f}s (± {np.std(nw_times):.3f}s)")
        print(f"   ► OVERHEAD   : +{overhead_pct:.2f}%")
        print(f"   └ [noWorkflow Break-down] Retrieval: {nw_retrieval:.3f}s | LLM Sintese: {nw_llm:.3f}s")
        print("-" * 60)

    # Sumário Global (Tabela 1 do Artigo)
    if total_py_mean and total_nw_mean:
        global_py = np.mean(total_py_mean)
        global_nw = np.mean(total_nw_mean)
        global_overhead = ((global_nw - global_py) / global_py) * 100
        
        print(f"MÉDIA GERAL DO EXPERIMENTO (TABELA 1 DO ARTIGO):")
        print(f"   Tempo Médio Python Puro: {global_py:.3f}s")
        print(f"   Tempo Médio noWorkflow : {global_nw:.3f}s")
        print(f"   OVERHEAD GLOBAL        : +{global_overhead:.2f}%")
        print("="*60)

if __name__ == "__main__":
    analyze()