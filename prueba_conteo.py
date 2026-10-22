"""Prueba sin front: pasa N objetos simulados por la cinta y verifica que cada uno se cuente UNA vez.

Uso:  ../.venv/bin/python prueba_conteo.py [N] [clip|falso]
"""
import json
import os
import sys
import tempfile
import time
from collections import Counter

from pipeline import AQUI, ClasificadorFalso, ClipZeroShot, FolderSource, Motor, Registro

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
modo = sys.argv[2] if len(sys.argv) > 2 else "falso"
clases = json.load(open(os.path.join(AQUI, "clases.json"), encoding="utf-8"))
clf = ClasificadorFalso(clases) if modo == "falso" else ClipZeroShot(clases)
while not clf.listo:
    if clf.error:
        sys.exit("error del modelo: " + clf.error)
    time.sleep(0.5)

reg = Registro(os.path.join(tempfile.mkdtemp(), "t.db"))
fuente = FolderSource(os.environ.get("PETVISION_SIM", os.path.join(AQUI, "simulacion")), clases=["bottle"])
motor = Motor(clases, clf, reg, "simulacion")

# el motor usa tiempo real (monotonic) para sostener la salida: se acelera con un reloj simulado
import pipeline

reloj = [0.0]
pipeline.time.monotonic = lambda: reloj[0]
t0 = time.time()
cuadros = 0
while not (fuente.objetos >= N and fuente.pista_actual is None and motor.estado == "VACIO"):
    reloj[0] += 1 / 8
    motor.paso(fuente.read(), fuente.pista_actual)
    cuadros += 1
    if cuadros > 5000:
        break

ev = reg.ultimos(1000)
print(f"objetos que pasaron: {fuente.objetos}   eventos contados: {len(ev)}   cuadros: {cuadros}   ({time.time()-t0:.0f}s)")
print("por clase detectada:", dict(Counter(e["clase"] for e in ev)))
print("por origen (pista):", dict(Counter(e["pista"] for e in ev)))
print("dobles/flags:", [(e["id"], e["flags"]) for e in ev if e["flags"]])
# montaje de revisión: recorte + clase predicha + origen (carpeta de Open Images), en data/crops/montaje.jpg
import cv2
import numpy as np
from pipeline import CROPS

celdas = []
for e in sorted(ev, key=lambda e: e["id"]):
    img = cv2.imread(os.path.join(CROPS, e["recorte"])) if e.get("recorte") else None
    if img is None:
        continue
    img = cv2.resize(img, (200, 200))
    cv2.rectangle(img, (0, 0), (200, 34), (0, 0, 0), -1)
    cv2.putText(img, f'{e["clase"]} {e["conf"]:.0%}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f'origen: {e["pista"]}', (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1, cv2.LINE_AA)
    celdas.append(img)
while len(celdas) % 6:
    celdas.append(np.zeros((200, 200, 3), np.uint8))
filas = [np.hstack(celdas[i:i + 6]) for i in range(0, len(celdas), 6)]
if filas:
    cv2.imwrite(os.path.join(CROPS, "montaje.jpg"), np.vstack(filas), [cv2.IMWRITE_JPEG_QUALITY, 80])
cruce = Counter((e["pista"], e["clase"]) for e in ev)
print("cruce origen x clase:", dict(cruce))
ok = len(ev) == fuente.objetos
print("RESULTADO:", "OK, ni dobles ni perdidos" if ok else "REVISAR: el conteo no coincide")
sys.exit(0 if ok else 1)
