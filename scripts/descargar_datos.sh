#!/usr/bin/env bash
#
# Restaura los datasets pesados que NO se versionan en git (*.zip > 100 MB).
# Completá las URLs de origen donde dice <PEGAR_URL> y corré:
#
#   bash scripts/descargar_datos.sh
#
# Los datasets de los Casos de Negocio provienen del Drive del curso; el de
# weatherAUS es público (Kaggle: "Rain in Australia").
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# formato: "ruta/destino/archivo.zip|URL_DE_DESCARGA"
DATASETS=(
  "02-desafio-profesional-etapa-1/casos-de-negocio/subtes/subtes.zip|<PEGAR_URL>"
  "02-desafio-profesional-etapa-1/casos-de-negocio/airbnb/airbnb.zip|<PEGAR_URL>"
  "02-desafio-profesional-etapa-1/casos-de-negocio/cambio-climatico/cambio-climatico.zip|<PEGAR_URL>"
  "02-desafio-profesional-etapa-1/casos-de-negocio/diabetes/diabetes.zip|<PEGAR_URL>"
  "01-introduccion-a-machine-learning/notebooks/ejercicio-7-xgboost/weatherAUS.csv.zip|<PEGAR_URL>"
)

descargar () {
  local dest="$ROOT/$1" url="$2"
  if [ -f "$dest" ]; then echo "  [OK]    ya existe: $1"; return; fi
  if [ "$url" = "<PEGAR_URL>" ]; then echo "  [SKIP]  falta URL para: $1"; return; fi
  echo "  [GET]   $1"
  mkdir -p "$(dirname "$dest")"
  curl -fSL "$url" -o "$dest"
}

echo "==> Restaurando datasets pesados (no versionados)"
for item in "${DATASETS[@]}"; do
  descargar "${item%%|*}" "${item##*|}"
done
echo "==> Listo. Recordá descomprimir los .zip donde corresponda."
