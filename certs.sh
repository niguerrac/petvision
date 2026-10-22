#!/bin/sh
# Genera certs/cert.pem y certs/key.pem para HTTPS.
# 1) Si Tailscale tiene HTTPS habilitado y el usuario puede pedir certificados: cert real de *.ts.net
#    (sin avisos en el navegador). Requiere UNA vez:  sudo tailscale set --operator=$USER
# 2) Si no: autofirmado con el hostname y las IPs de esta máquina (el navegador pide aceptar la excepción).
cd "$(dirname "$0")" || exit 1
mkdir -p certs

FQDN=$(tailscale status --json 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null)
if [ -n "$FQDN" ] && tailscale cert --cert-file certs/cert.pem --key-file certs/key.pem "$FQDN" >/dev/null 2>&1; then
  echo "Certificado de Tailscale: https://$FQDN:${PORT:-8443}"
  exit 0
fi

SAN="DNS:$(hostname),DNS:localhost,IP:127.0.0.1"
for ip in $(ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep -v '^127\.'); do SAN="$SAN,IP:$ip"; done
[ -n "$FQDN" ] && SAN="$SAN,DNS:$FQDN"
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=petvision" -addext "subjectAltName=$SAN" >/dev/null 2>&1 && echo "Certificado autofirmado ($SAN)"
chmod 600 certs/key.pem
