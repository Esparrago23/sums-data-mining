#!/bin/bash
# retrain_real.sh
# ================
# Reentrena el modelo de riesgo (Subcomponente B) con las cédulas REALES
# capturadas en sums-API, pensado para correrse una vez al mes por cron en
# el HOST del servidor (no dentro del contenedor).
#
# SEGURIDAD DEL DESPLIEGUE: solo reinicia la API (para cargar el modelo
# nuevo) si el reentrenamiento terminó en verde -- es decir, si pasaron los
# asserts de cordura de run_all.py (los 3 modelos con accuracy > 0.5,
# macro-F1 del ganador > 0.5, lista priorizada no vacía y ordenada). Si algo
# salió mal (poca data, una racha de cédulas mal capturadas, etc.), el
# modelo actual sigue sirviendo sin interrupción y el error queda en el log
# para revisar a mano -- nunca se despliega un modelo nuevo sin pasar ese
# filtro mínimo.
#
# Uso manual:
#   ./retrain_real.sh
#
# Uso en crontab (mensual, día 1 de cada mes a las 3:00am; ajusta la ruta):
#   0 3 1 * * /ruta/absoluta/a/sums-data-mining/retrain_real.sh
#
# (No hace falta redirigir la salida a mano -- el script ya escribe su
# propio log con fecha en logs/retrain_YYYY-MM.log.)

set -u
# pipefail: sin esto, el `| tee` de abajo hace que el script SIEMPRE salga
# con el exit code de `tee` (0), aunque el reentrenamiento haya fallado --
# se detectó justo así al probarlo (el script no reiniciaba la API, que es
# lo importante, pero reportaba exit 0 de todos modos). Con pipefail, el
# exit code real del pipeline es el del primer comando que falle.
set -o pipefail
cd "$(dirname "$0")" || exit 1

FECHA_HORA=$(date '+%Y-%m-%d %H:%M:%S')
MES=$(date '+%Y-%m')
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/retrain_$MES.log"
mkdir -p "$LOG_DIR"

{
  echo "===================================================================="
  echo "[$FECHA_HORA] Iniciando reentrenamiento mensual (--fuente real)"
  echo "===================================================================="

  if docker exec sums_mineria_api python subcomponente_B_ETL_Risk/src/run_all.py --fuente real; then
    echo ""
    echo "[$FECHA_HORA] Reentrenamiento OK -- reiniciando mineria-api para cargar el modelo nuevo."
    if docker compose restart mineria-api; then
      echo "[$FECHA_HORA] Listo. mineria-api reiniciada con el modelo reentrenado."
    else
      echo "[$FECHA_HORA] *** El reentrenamiento salió bien pero 'docker compose restart' falló -- revisa Docker a mano. ***"
      exit 1
    fi
  else
    echo ""
    echo "[$FECHA_HORA] *** Reentrenamiento FALLÓ (ver salida arriba) -- NO se reinicia la API, sigue sirviendo el modelo anterior sin interrupción. ***"
    echo "[$FECHA_HORA] Causas típicas: pocas cédulas no-borrador en la BD, o error de conexión (revisa SUMS_DB_* en .env)."
    exit 1
  fi
} 2>&1 | tee -a "$LOG_FILE"
