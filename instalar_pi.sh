#!/bin/sh
# Instala PetVision en una Raspberry Pi (Raspberry Pi OS de 64 bits, Bookworm o posterior).
# Se ejecuta DENTRO de la Pi, desde la carpeta ya copiada:   cd ~/petvision && ./instalar_pi.sh
set -e
cd "$(dirname "$0")"
AQUI=$(pwd)
USUARIO=$(id -un)

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: este sistema es de $(uname -m). PyTorch necesita Raspberry Pi OS de 64 bits (aarch64)." >&2
  echo "Vuelve a grabar la tarjeta con 'Raspberry Pi OS Lite (64-bit)'." >&2
  exit 1
fi

echo "== 1/5 Paquetes del sistema (numpy, OpenCV y picamera2 vienen del sistema)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3-venv python3-pip python3-numpy python3-opencv python3-picamera2 openssl
sudo usermod -aG video "$USUARIO"

echo "== 2/5 Entorno Python (hereda los paquetes del sistema)"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install fastapi "uvicorn[standard]" pillow open_clip_torch torch torchvision

echo "== 3/5 Modelo CLIP"
if ls "$HOME"/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots/*/open_clip_model.safetensors >/dev/null 2>&1; then
  echo "Modelo ya está en caché."
else
  echo "El modelo no está en caché: intento descargarlo (~600 MB, puede tardar bastante)."
  echo "Es más rápido copiarlo desde otro equipo (ver README, recuadro Modelo sin descargar en la Pi)."
  HF_HUB_DISABLE_XET=1 .venv/bin/python bajar_modelo.py || echo "AVISO: no se pudo descargar el modelo; el servicio avisará 'Error del modelo' hasta que esté."
fi

echo "== 4/5 Certificado HTTPS"
[ -f certs/cert.pem ] || ./certs.sh

echo "== 5/5 Servicio systemd (arranca solo al encender la Pi)"
sudo tee /etc/systemd/system/petvision.service >/dev/null <<EOF
[Unit]
Description=PetVision (clasificacion y conteo en cinta)
After=network-online.target
Wants=network-online.target

[Service]
User=$USUARIO
SupplementaryGroups=video
WorkingDirectory=$AQUI
Environment=PETVISION_FUENTE=picamera
Environment=PETVISION_ASYNC=1
Environment=PETVISION_HILOS=3
Environment=PETVISION_FPS=6
ExecStart=$AQUI/run.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now petvision
sleep 8
systemctl --no-pager --lines=8 status petvision || true

IP=$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)
echo
echo "Listo. Abre en el navegador:  https://$(hostname).local:8443/   o   https://$IP:8443/"
echo "Ver el registro:  journalctl -u petvision -f     Detener: sudo systemctl stop petvision"
echo "Cámara: si el panel avisa un error de fuente, revisa 'rpicam-hello --list-cameras' y el cable plano."
