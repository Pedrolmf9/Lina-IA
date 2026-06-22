#!/usr/bin/env bash
# Bateria de experimentos — 5 perguntas × 10 execuções × 2 grupos = 100 runs
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH=src

PERGUNTAS=(
  "Quem desenvolveu a Lina?"
  "Como posso combinar alimentos para evitar picos de glicemia?"
  "Quais são os sintomas de hipoglicemia?"
  "Como a Lina protege minha privacidade?"
  "Quais medicamentos para diabetes a Lina pode me lembrar de tomar?"
)

REPETICOES=10
RUNNER=${1:-"python"}   # "python" ou "noworkflow"
SCRIPT="src/retrieval/retrieve-queryengine-PROV.py"

# EXPORTA A VARIÁVEL PARA O PYTHON LER
export RUN_MODE=$RUNNER

echo "=== Iniciando bateria: RUN_MODE=$RUN_MODE, REPETICOES=$REPETICOES ==="

for pergunta in "${PERGUNTAS[@]}"; do
  echo "--- Pergunta: $pergunta ---"
  for i in $(seq 1 $REPETICOES); do
    echo "  Execução $i/$REPETICOES"
    if [ "$RUN_MODE" = "noworkflow" ]; then
      now run --dir . "$SCRIPT" "$pergunta" > /dev/null 2>&1
    else
      python "$SCRIPT" "$pergunta" > /dev/null 2>&1
    fi
    sleep 2  # evitar rate limit da OpenAI
  done
done

echo "=== Bateria concluída. Runs salvos em data/provenance_logs/ ==="