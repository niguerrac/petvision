#!/bin/sh
# Arranca PetVision por HTTPS.  Uso:  ./run.sh   (opcional: PORT=8443 PETVISION_FUENTE=webcam|simulacion PETVISION_TOKEN=clave)
cd "$(dirname "$0")" || exit 1
[ -f certs/cert.pem ] || ./certs.sh
# con el modelo ya en caché no hace falta internet (la reunión puede no tener red); la 1ª vez: HF_HUB_OFFLINE=0 ./run.sh
[ -d "$HOME/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai" ] && export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PY=".venv/bin/python"                      # entorno propio (instalación en la Pi)
[ -x "$PY" ] || PY="../.venv/bin/python"   # entorno del proyecto (desarrollo en el nodo)
[ -x "$PY" ] || PY="python3"
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8443}" \
  --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
