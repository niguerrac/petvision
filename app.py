"""PetVision — servidor FastAPI: video en vivo (MJPEG), API de eventos/estadísticas y el front.

Arranque (HTTPS): ./run.sh        Variables: PETVISION_FUENTE=webcam|picamera|simulacion  PETVISION_TOKEN=clave
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import shutil
import threading
import time
from datetime import datetime

import cv2
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse

from pipeline import (AQUI, CROPS, DATA, ClasificadorFalso, ClipZeroShot, FolderSource, Motor, PiCamSource, Registro,
                      WebcamSource)

TOKEN = os.environ.get("PETVISION_TOKEN", "")
DATASET_SIM = os.environ.get("PETVISION_SIM", os.path.join(AQUI, "simulacion"))  # carpeta con fotos para el modo simulación
CLASES_SIM = [c for c in os.environ.get("PETVISION_SIM_CLASES", "").split(",") if c]
FPS_PROCESO = float(os.environ.get("PETVISION_FPS", "8"))
ASINC = os.environ.get("PETVISION_ASYNC", "1") == "1"  # clasificar en un hilo aparte

with open(os.path.join(AQUI, "clases.json"), encoding="utf-8") as f:
    CLASES = json.load(f)


def autorizar(request: Request):
    if not TOKEN:
        return
    t = request.query_params.get("t") or request.headers.get("x-token") or ""
    if not hmac.compare_digest(t, TOKEN):
        raise HTTPException(401, "No autorizado")


app = FastAPI(title="PetVision", dependencies=[Depends(autorizar)])


class Servicio:
    """Hilo que lee la fuente, corre el motor y deja el último cuadro anotado como JPEG."""

    def __init__(self):
        self.registro = Registro(os.path.join(DATA, "eventos.db"))
        if os.environ.get("PETVISION_CLASIFICADOR") == "falso":
            self.clf = ClasificadorFalso(CLASES, float(os.environ.get("PETVISION_RETARDO", "0")))
        else:
            self.clf = ClipZeroShot(CLASES)
        self.fuente = None
        self.motor = None
        self.jpeg: bytes | None = None
        self.fps = 0.0
        self.error_fuente = None
        self._run = True
        self._lock = threading.Lock()
        self.cambiar_fuente(os.environ.get("PETVISION_FUENTE", "webcam"))
        threading.Thread(target=self._bucle, daemon=True).start()

    def cambiar_fuente(self, tipo: str, indice: int = 0):
        with self._lock:
            nueva = None
            self.error_fuente = None
            try:
                if tipo == "webcam":
                    nueva = WebcamSource(indice)
                elif tipo == "picamera":
                    nueva = PiCamSource()
                else:
                    nueva = FolderSource(DATASET_SIM, clases=CLASES_SIM)
            except Exception as e:  # noqa: BLE001
                self.error_fuente = f"{type(e).__name__}: {e}"
                try:
                    nueva = FolderSource(DATASET_SIM, clases=CLASES_SIM)
                except Exception as e2:  # noqa: BLE001
                    self.error_fuente += f" | simulación: {e2}"
            if self.fuente is not None:
                self.fuente.close()
            self.fuente = nueva
            cfg = self.motor.cfg if self.motor else None
            self.motor = Motor(CLASES, self.clf, self.registro, nueva.nombre if nueva else "ninguna", asincrono=ASINC)
            if cfg:
                self.motor.cfg.update(cfg)

    def _bucle(self):
        t_prev = time.monotonic()
        while self._run:
            t0 = time.monotonic()
            with self._lock:
                fuente, motor = self.fuente, self.motor
                frame = fuente.read() if fuente else None
                if frame is not None:
                    anotado = motor.paso(frame, getattr(fuente, "pista_actual", None))
                    ok, buf = cv2.imencode(".jpg", anotado, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.jpeg = buf.tobytes()
            ahora = time.monotonic()
            self.fps = 0.9 * self.fps + 0.1 / max(1e-3, ahora - t_prev) if self.fps else 1.0 / max(1e-3, ahora - t_prev)
            t_prev = ahora
            time.sleep(max(0.0, 1.0 / FPS_PROCESO - (time.monotonic() - t0)))


servicio: Servicio | None = None


@app.on_event("startup")
def _inicio():
    global servicio
    servicio = Servicio()


# --------------------------------------------------------------------------- páginas y video
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(AQUI, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/video")
async def video():
    async def gen():
        ultimo = None
        while True:
            jpg = servicio.jpeg
            if jpg is not None and jpg is not ultimo:
                ultimo = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/crops/{nombre}")
def recorte(nombre: str):
    ruta = os.path.join(CROPS, os.path.basename(nombre))
    if not os.path.exists(ruta):
        raise HTTPException(404)
    return FileResponse(ruta, media_type="image/jpeg")


# --------------------------------------------------------------------------- API
@app.get("/api/clases")
def clases():
    return [{k: c[k] for k in ("id", "nombre", "color") if k in c} | {"alerta": bool(c.get("alerta"))} for c in CLASES]


@app.get("/api/estado")
def estado():
    s = dict(servicio.motor.snap)
    s["fps"] = round(servicio.fps, 1)
    s["error_fuente"] = servicio.error_fuente
    s["ultimo_evento"] = servicio.motor.ultimo_evento
    s["objetos_simulados"] = getattr(servicio.fuente, "objetos", None)
    return s


@app.get("/api/eventos")
def eventos(limite: int = 50):
    return servicio.registro.ultimos(max(1, min(500, limite)))


@app.get("/api/estadisticas")
def estadisticas():
    return servicio.registro.estadisticas()


@app.get("/api/eventos.csv")
def eventos_csv():
    return Response(servicio.registro.csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=petvision_eventos.csv"})


@app.post("/api/calibrar")
def calibrar():
    servicio.motor.calibrar()
    return {"ok": True}


@app.post("/api/fuente")
def fuente(datos: dict = Body(...)):
    tipo = datos.get("tipo", "simulacion")
    if tipo not in ("webcam", "picamera", "simulacion"):
        raise HTTPException(400, "tipo debe ser webcam, picamera o simulacion")
    servicio.cambiar_fuente(tipo, int(datos.get("indice", 0)))
    return {"ok": True, "fuente": servicio.motor.fuente_nombre, "error": servicio.error_fuente}


@app.post("/api/config")
def config(datos: dict = Body(...)):
    cfg = servicio.motor.cfg
    if "roi" in datos:
        r = [float(v) for v in datos["roi"]]
        if len(r) != 4 or any(v < 0 or v > 1 for v in r) or r[0] + r[2] > 1 or r[1] + r[3] > 1 or r[2] < 0.05 or r[3] < 0.05:
            raise HTTPException(400, "roi inválida")
        cfg["roi"] = r
    for k in ("umbral_pix", "entrada", "salida", "min_conf", "sostener_salida_s"):
        if k in datos:
            cfg[k] = float(datos[k])
    return cfg


@app.post("/api/reiniciar")
def reiniciar(datos: dict = Body(...)):
    if datos.get("confirmar") is not True:
        raise HTTPException(400, "falta confirmar")
    ruta = servicio.registro.ruta
    respaldo = os.path.join(DATA, f"respaldo_{datetime.now():%Y%m%d_%H%M%S}.db")
    shutil.copy(ruta, respaldo)
    servicio.registro.vaciar()
    servicio.motor.ultimo_evento = None
    return {"ok": True, "respaldo": os.path.basename(respaldo)}
