# 🤖 LINA IA — Intelligent Assistant Framework

**LINA IA** é uma plataforma modular para o desenvolvimento de **assistentes conversacionais inteligentes**, criada por estudantes da **Universidade Federal Fluminense (UFF)**.  
O projeto combina **LLMs**, **memória conversacional**, e **busca vetorial (RAG)** para oferecer experiências personalizadas e seguras — voltada ao apoio de pessoas com diabetes.

---

## 🚀 Visão Geral

LINA IA permite criar agentes inteligentes com:

- **LLMs configuráveis** (OpenAI GPT-4/GPT-3.5)
- **RAG com Qdrant** para busca semântica eficiente
- **Memória persistente via Redis**
- **Prompts de sistema customizáveis, utilizando o Framework LLAMAINDEX**
- **Ferramentas (Tools)** que ampliam as capacidades dos agentes
- **Suporte a múltiplos usuários**, com isolamento de contexto

---

## ⚙️ Como Executar o Projeto

Siga o passo a passo abaixo para configurar e rodar o assistente conversacional localmente utilizando um ambiente virtual Python (`venv`).

### 1. Pré-requisitos
- **Python 3.10+** instalado
- **Docker** e **Docker Compose** instalados (para subir o Qdrant e o Redis)
- Chave de API da **OpenAI**

### 2. Configurar o Ambiente Virtual (venv)
É altamente recomendado isolar as dependências do projeto num ambiente virtual para evitar conflitos de bibliotecas.

```bash
# Navegue até a pasta do projeto
cd Lina-IA

# Crie o ambiente virtual
python3 -m venv venv

# Ative o ambiente virtual
# No Linux/MacOS/WSL:
source venv/bin/activate
# No Windows (PowerShell):
 .\venv\Scripts\Activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente
O projeto precisa de chaves de API e caminhos que ficam no arquivo `.env`.

```bash
# Copie o arquivo de template
cp .env.template .env
```
Abra o `.env` no seu editor de código e cole a sua `OPEN_API_KEY`.

### 4. Subir a Infraestrutura Base (Docker)
A memória do chat e o banco vetorial rodam em containers via Docker Compose.

```bash
# Subir os containers em background
docker compose up -d
```

> **Nota:** O dashboard do Qdrant pode ser acessado no navegador através de [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### 5. Ingestão de Dados (Obrigatório na 1ª Vez)
Como o banco vetorial inicializa vazio, você precisa preenchê-lo e criar a coleção `lina_docs_tb`. Coloque seus arquivos `.pdf` para formar a base de conhecimento (RAG) na pasta configurada em `TB_DOCS_PATH` (por padrão, `data/raw/`) e rode a indexação:

```bash
./scripts/ingest-tb.sh
```

### 6. Iniciar a Interação (Chat)
Com os containers ativos e o ambiente configurado, rode o script do chat engine para conversar com a Lina diretamente no terminal:

```bash
./scripts/chatengine-tb.sh
```

*(Para fazer consultas rápidas no banco vetorial sem o histórico de chat, você também pode usar `./scripts/queryengine.sh "Sua Pergunta"`).*

---

## 🔬 Proveniência e Reprodutibilidade (E-Science)

Esta seção descreve a camada de rastreabilidade científica do projeto, desenvolvida como parte da disciplina de E-Science da UFF. O objetivo é capturar a linhagem completa do pipeline RAG — da ingestão de documentos até a geração de respostas — garantindo auditabilidade e reprodutibilidade dos experimentos.

### Arquitetura de Instrumentação

A captura de proveniência opera em três camadas complementares:

| Camada | Ferramenta | O que captura |
|---|---|---|
| **1 — Semântica** | LlamaIndex CallbackManager | `node_ids`, scores de similaridade, tokens, latência por etapa |
| **2 — Experimentos** | MLflow (backend SQLite) | Parâmetros prospectivos, métricas retrospectivas, artefatos JSON |
| **3 — Software** | noWorkflow | Grafo de chamadas Python, dependências de arquivos, `trial_id` |

Cada execução instrumentada gera um artefato JSON em `data/provenance_logs/` e um run registrado no MLflow, com o `trial_id` do noWorkflow vinculado via tag — fechando o grafo de linhagem end-to-end.

### Pré-requisitos Adicionais

```bash
# noWorkflow (instalado via requirements.txt)
# Verifique a instalação:
now --version

# MLflow UI (opcional, para visualizar os runs)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Acesse em: http://localhost:5000
```

> ⚠️ **Atenção a conflitos de dependência:** o noWorkflow requer `SQLAlchemy<=1.4`, enquanto versões recentes do LlamaIndex exigem `>=2.0`. O `requirements.txt` já incluímos o pin correto (`SQLAlchemy>=1.4.49,<2.0.0`). Não altere essa versão.

### Ingestão com Proveniência

Para registrar a ingestão de documentos no MLflow (gera `data/provenance_logs/ingest_*.json`):

```bash
PYTHONPATH=src python src/ingestion/ingest-tb-PROV.py
```

Anote o `mlflow_run_id` exibido no log — ele identifica a versão do índice vetorial usado nos experimentos.

```bash
# Exporte para vincular a ingestão às consultas subsequentes
export INGEST_RUN_ID="<run_id anotado>"
```

### Rodando a Bateria de Experimentos

Os experimentos comparam o pipeline em dois modos: **Python puro** (controle) e **noWorkflow** (instrumentado). São 5 perguntas × 10 repetições × 2 grupos = **100 execuções**.

> 💡 **Recomendado:** execute dentro de um `tmux` ou `screen` — a bateria leva aproximadamente 35 minutos no total.

**Passo 1 — Grupo controle (Python puro):**
```bash
bash experiments/run_battery.sh python
```

**Passo 2 — Grupo experimental (noWorkflow):**
```bash
bash experiments/run_battery.sh noworkflow
```

**Passo 3 — Analisar os resultados:**
```bash
PYTHONPATH=src python experiments/analyze_results.py
```

O script de análise remove automaticamente o maior e o menor valor de cada célula (remoção de outliers), exibe a tabela de overhead por pergunta e o breakdown de latência por etapa (Retrieval vs. Síntese LLM).

### Consulta isolada com proveniência

Para rodar uma única pergunta com rastreamento completo:

```bash
# Modo Python puro
RUN_MODE=python PYTHONPATH=src python src/retrieval/retrieve-queryengine-PROV.py "Sua pergunta aqui"

# Modo noWorkflow
RUN_MODE=noworkflow PYTHONPATH=src now run src/retrieval/retrieve-queryengine-PROV.py "Sua pergunta aqui"
```

### Estrutura dos artefatos gerados

```
data/
└── provenance_logs/
    ├── ingest_<run_id>_<timestamp>.json   # Linhagem da ingestão
    └── query_<trial_id>_<timestamp>.json  # Linhagem de cada consulta

mlflow.db                                  # Banco de runs do MLflow
.noworkflow/
└── db.sqlite                              # Trials do noWorkflow
```

Cada JSON de consulta contém:

```jsonc
{
  "experiment": "Validacao_RAG_CLI",
  "mlflow_run_id": "...",
  "noworkflow_trial_id": "...",      // null se executado sem now run
  "prospect_parameters": {           // O que foi configurado ANTES
    "llm_model": "gpt-4o-mini",
    "similarity_top_k": 5,
    "git_commit": "83b9f65",
    "runner": "noworkflow"
  },
  "retrospect_execution": {          // O que aconteceu DURANTE
    "user_query": "...",
    "retrieval_events": [...],       // node_ids, scores, trechos recuperados
    "llm_events": [...],             // resposta gerada, tokens consumidos
    "retrieval_latency_seconds": 1.0,
    "llm_latency_seconds": 2.1,
    "wall_time_seconds": 3.1
  }
}
```

### Resultados do experimento

Os experimentos realizados evidenciaram:

- **Overhead médio global:** +679% introduzido pelo noWorkflow, concentrado em ~97% na etapa de recuperação vetorial
- **Determinismo:** retrieval 100% idêntico entre repetições (mesmos `node_ids` e scores) em 4 das 5 perguntas, confirmando reprodutibilidade com `temperature=0.0`
- **Diagnóstico de falha:** identificação de um falso positivo semântico (score 0.79) na pergunta sobre hipoglicemia — falha invisível sem instrumentação, rastreável com ela
- **Custo financeiro:** 38.536 tokens totais, ~$0,02 — o overhead é computacional, não de
