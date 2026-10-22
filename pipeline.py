"""Núcleo de PetVision: fuentes de video, presencia en la cinta, clasificación y conteo sin dobles.

Un objeto se cuenta UNA vez: máquina de estados sobre una ROI fija.
    VACIO -> ENTRANDO -> PRESENTE -> SALIENDO -> (evento único) -> VACIO
El evento se emite al salir, con el promedio de probabilidades de todos los cuadros.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import queue
import random
import sqlite3
import threading
import time
from datetime import datetime

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PETVISION_DATA", os.path.join(AQUI, "data"))
CROPS = os.path.join(DATA, "crops")
ANCHO = 640
os.makedirs(CROPS, exist_ok=True)


# --------------------------------------------------------------------------- fuentes
def _a_ancho(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == ANCHO:
        return frame
    return cv2.resize(frame, (ANCHO, int(h * ANCHO / w)))


class WebcamSource:
    nombre = "webcam"
    pista_actual = None

    def __init__(self, indice: int = 0):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir /dev/video{indice}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._lock = threading.Lock()
        self._run = True
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    def _leer(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f
            else:
                time.sleep(0.05)

    def read(self):
        with self._lock:
            f = self._frame
        return None if f is None else _a_ancho(f.copy())

    def close(self):
        self._run = False
        self._hilo.join(1)
        self.cap.release()


class PiCamSource:
    """Cámara oficial de Raspberry Pi (conector CSI) con picamera2.

    Solo funciona en la Pi (paquete del sistema: sudo apt install python3-picamera2).
    """

    nombre = "picamera"
    pista_actual = None

    def __init__(self):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(
            main={"size": (ANCHO, 480), "format": "RGB888"}, controls={"FrameRate": 15}
        )
        self.cam.configure(cfg)
        self.cam.start()

    def read(self):
        # el formato "RGB888" de picamera2 guarda los píxeles en orden B,G,R: ya sirve para OpenCV
        return self.cam.capture_array("main")

    def close(self):
        self.cam.stop()
        self.cam.close()


class FolderSource:
    """Simula la cinta: imágenes que cruzan de izquierda a derecha sobre un fondo de cinta fijo."""

    nombre = "simulacion"
    ALTO = 480
    HUECO = 16  # cuadros de cinta vacía entre objetos
    PASOS = 38  # cuadros que tarda un objeto en cruzar

    def __init__(self, raiz: str, semilla: int = 7, clases: list[str] | None = None):
        self.rnd = random.Random(semilla)
        self.grupos = self._listar(raiz)
        if clases:  # filtra por nombre de subcarpeta (p. ej. ["bottle"]); si no coincide ninguna, usa todas
            self.grupos = {k: v for k, v in self.grupos.items() if k in clases} or self.grupos
        if not self.grupos:
            raise RuntimeError(f"No hay imágenes en {raiz}")
        self._ciclo = itertools.cycle(sorted(self.grupos))
        self.cinta = self._fondo()
        self.pista_actual = None
        self.objetos = 0  # cuántos objetos entraron a la cinta (verdad para contrastar el conteo)
        self._gen = self._generar()

    def _listar(self, raiz):
        grupos: dict[str, list] = {}
        resumen = os.path.join(raiz, "resumen.csv")
        if os.path.exists(resumen):
            with open(resumen, newline="") as f:
                for r in csv.DictReader(f):
                    ruta = os.path.join(raiz, r["clase"], r["ImageID"] + ".jpg")
                    if os.path.exists(ruta):
                        box = (float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"]))
                        grupos.setdefault(r["clase"], []).append((ruta, box))
        else:
            for dp, _, fs in os.walk(raiz):
                for n in fs:
                    if n.lower().endswith((".jpg", ".jpeg", ".png")):
                        grupos.setdefault(os.path.basename(dp), []).append((os.path.join(dp, n), None))
        for g in grupos.values():
            self.rnd.shuffle(g)
        self._idx = {k: 0 for k in grupos}
        return grupos

    def _fondo(self):
        rng = np.random.default_rng(3)
        base = np.full((self.ALTO, ANCHO, 3), (92, 92, 96), np.uint8)
        base = base.astype(np.int16) + rng.normal(0, 3, base.shape[:2])[..., None].astype(np.int16)
        base[::48] -= 14  # juntas de la cinta
        base[:, :6] -= 25
        base[:, -6:] -= 25
        return np.clip(base, 0, 255).astype(np.uint8)

    def _siguiente(self):
        clase = next(self._ciclo)
        lista = self.grupos[clase]
        i = self._idx[clase] % len(lista)
        self._idx[clase] += 1
        return clase, lista[i]

    def _objeto(self, ruta, box):
        img = cv2.imread(ruta)
        if img is None:
            return None
        if box:
            h, w = img.shape[:2]
            x0, y0, x1, y1 = box
            mx, my = (x1 - x0) * 0.12, (y1 - y0) * 0.12
            img = img[int(max(0, y0 - my) * h):int(min(1, y1 + my) * h), int(max(0, x0 - mx) * w):int(min(1, x1 + mx) * w)]
        if img.size == 0:
            return None
        h, w = img.shape[:2]
        esc = min(self.ALTO * 0.55 / h, ANCHO * 0.45 / w)
        return cv2.resize(img, (max(8, int(w * esc)), max(8, int(h * esc))))

    def _generar(self):
        while True:
            self.pista_actual = None
            for _ in range(self.HUECO):
                yield self.cinta.copy()
            clase, (ruta, box) = self._siguiente()
            obj = self._objeto(ruta, box)
            if obj is None:
                continue
            self.objetos += 1
            self.pista_actual = clase
            oh, ow = obj.shape[:2]
            y0 = max(0, min(self.ALTO - oh, (self.ALTO - oh) // 2 + self.rnd.randint(-12, 12)))
            for i in range(self.PASOS):
                x = int(-ow + (ANCHO + ow) * i / (self.PASOS - 1))
                f = self.cinta.copy()
                a, b = max(0, x), min(ANCHO, x + ow)
                if b > a:
                    f[y0:y0 + oh, a:b] = obj[:, a - x:b - x]
                yield f

    def read(self):
        return next(self._gen)

    def close(self):
        pass


# --------------------------------------------------------------------------- clasificadores
def _cuadrar(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    d = abs(h - w)
    if h > w:
        return cv2.copyMakeBorder(img, 0, 0, d // 2, d - d // 2, cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(img, d // 2, d - d // 2, 0, 0, cv2.BORDER_REPLICATE)


class ClipZeroShot:
    """Clasifica con CLIP sin entrenar, comparando contra descripciones de cada clase."""

    def __init__(self, clases: list[dict], modelo: str = "ViT-B-32-quickgelu", pretrained: str = "openai"):
        self.clases = clases
        self.modelo = modelo
        self.pretrained = pretrained
        self.listo = False
        self.error = None
        threading.Thread(target=self._cargar, daemon=True).start()

    def _cargar(self):
        try:
            import open_clip
            import torch

            torch.set_num_threads(int(os.environ.get("PETVISION_HILOS", "0")) or max(1, (os.cpu_count() or 4) // 2))
            self.torch = torch
            self.model, _, self.pre = open_clip.create_model_and_transforms(self.modelo, pretrained=self.pretrained)
            self.model.eval()
            tok = open_clip.get_tokenizer(self.modelo)
            with torch.no_grad():
                feats = []
                for c in self.clases:
                    t = self.model.encode_text(tok(c["prompts"]))
                    t = t / t.norm(dim=-1, keepdim=True)
                    m = t.mean(0)
                    feats.append(m / m.norm())
                self.txt = torch.stack(feats)
            self.listo = True
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    def predecir(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = cv2.cvtColor(_cuadrar(bgr), cv2.COLOR_BGR2RGB)
        img = self.pre(Image.fromarray(rgb)).unsqueeze(0)
        with self.torch.no_grad():
            f = self.model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            return (100.0 * f @ self.txt.T).softmax(-1)[0].numpy()


class ClasificadorFalso:
    """Para probar el conteo sin modelo: reparte las clases según el nombre de la carpeta de origen."""

    listo = True
    error = None

    def __init__(self, clases, retardo: float = 0.0):
        self.n = len(clases)
        self.retardo = retardo  # simula un clasificador lento (p. ej. CLIP en una Pi)

    def predecir(self, bgr):
        if self.retardo:
            time.sleep(self.retardo)
        p = np.full(self.n, 0.02)
        p[0] = 0.9
        return p / p.sum()


# --------------------------------------------------------------------------- registro
class Registro:
    def __init__(self, ruta: str):
        self.ruta = ruta
        self.db = sqlite3.connect(ruta, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS eventos(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, clase TEXT, conf REAL,
                    n_cuadros INTEGER, n_clasif INTEGER, duracion REAL, alerta INTEGER,
                    flags TEXT, recorte TEXT, probs TEXT, fuente TEXT, pista TEXT)"""
            )
            self.db.commit()

    def agregar(self, **e) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO eventos(ts,clase,conf,n_cuadros,n_clasif,duracion,alerta,flags,probs,fuente,pista)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (e["ts"], e["clase"], e["conf"], e["n_cuadros"], e["n_clasif"], e["duracion"], e["alerta"],
                 e["flags"], json.dumps(e["probs"]), e["fuente"], e.get("pista")),
            )
            self.db.commit()
            return cur.lastrowid

    def poner_recorte(self, id_: int, nombre: str):
        with self.lock:
            self.db.execute("UPDATE eventos SET recorte=? WHERE id=?", (nombre, id_))
            self.db.commit()

    def _filas(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def ultimos(self, n: int = 50):
        filas = self._filas("SELECT * FROM eventos ORDER BY id DESC LIMIT ?", (n,))
        for f in filas:
            f["probs"] = json.loads(f["probs"] or "{}")
        return filas

    def estadisticas(self):
        por_clase = self._filas("SELECT clase, COUNT(*) n, AVG(conf) conf FROM eventos GROUP BY clase")
        tot = self._filas("SELECT COUNT(*) n, COALESCE(SUM(alerta),0) alertas FROM eventos")[0]
        minutos = self._filas(
            "SELECT substr(ts,1,16) minuto, COUNT(*) n FROM eventos GROUP BY minuto ORDER BY minuto DESC LIMIT 30"
        )
        return {"total": tot["n"], "alertas": tot["alertas"], "por_clase": por_clase, "por_minuto": minutos[::-1]}

    def csv(self) -> str:
        filas = self._filas("SELECT id,ts,clase,conf,n_cuadros,n_clasif,duracion,alerta,flags,fuente,pista FROM eventos ORDER BY id")
        import io

        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["id", "ts", "clase", "conf", "n_cuadros", "n_clasif", "duracion_s", "alerta_vidrio", "flags", "fuente", "pista"])
        for f in filas:
            w.writerow(f.values())
        return out.getvalue()

    def vaciar(self):
        with self.lock:
            self.db.execute("DELETE FROM eventos")
            self.db.commit()


# --------------------------------------------------------------------------- motor
VACIO, ENTRANDO, PRESENTE, SALIENDO, RETENIDO, CALIBRANDO = "VACIO", "ENTRANDO", "PRESENTE", "SALIENDO", "RETENIDO", "CALIBRANDO"
COLOR = {VACIO: (160, 160, 160), ENTRANDO: (0, 215, 255), PRESENTE: (80, 200, 80), SALIENDO: (0, 140, 255),
         RETENIDO: (60, 60, 220), CALIBRANDO: (255, 160, 60)}

CONFIG_BASE = {
    "roi": [0.25, 0.10, 0.50, 0.80],  # x, y, ancho, alto normalizados
    "umbral_pix": 28,
    "entrada": 0.07,
    "salida": 0.025,
    "k_entrada": 2,
    "sostener_salida_s": 1.0,
    "min_conf": 0.50,
    "clasificar_cada_s": 0.30,
    "max_presente_s": 25.0,
    "calib_frames": 10,
}


class Motor:
    def __init__(self, clases, clasificador, registro: Registro, fuente_nombre: str, asincrono: bool = False):
        self.clases = clases
        self.ids = [c["id"] for c in clases]
        self.clf = clasificador
        self.reg = registro
        self.cfg = dict(CONFIG_BASE)
        self.fuente_nombre = fuente_nombre
        self.lock = threading.Lock()
        self.k3 = np.ones((3, 3), np.uint8)
        self.k15 = np.ones((15, 15), np.uint8)
        self.ultimo_evento = None
        self.en_vivo = None
        self.snap = {"estado": CALIBRANDO, "presencia": 0.0, "fuente": fuente_nombre, "modelo_listo": False,
                     "modelo_error": None, "en_vivo": None, "cfg": dict(self.cfg)}
        self.estado = VACIO
        self._gen = 0  # número de objeto: una lectura tardía de un objeto anterior se descarta
        self._pendientes = 0
        self._cv = threading.Condition()
        self.asinc = asincrono
        if asincrono:
            self._cola = queue.Queue()
            threading.Thread(target=self._trabajador, daemon=True).start()
        self.calibrar()

    # -- estado
    def calibrar(self):
        self._calib = []
        self._calentar = self.cfg.get("calentar_frames", 20)  # descarta los primeros cuadros: la cámara ajusta exposición
        self.bg = None
        self._reiniciar_objeto()
        self.estado = CALIBRANDO

    def _reiniciar_objeto(self):
        self._gen += 1
        self._k = 0
        self._suma = np.zeros(len(self.ids))
        self._peso = 0.0
        self._n_clasif = 0
        self._n_cuadros = 0
        self._t_ini = 0.0
        self._t_clasif = 0.0
        self._t_salida = 0.0
        self._mejor = (0.0, None)
        self._dobles = 0
        self._pista = None
        self.en_vivo = None

    def _roi_px(self, shape):
        h, w = shape[:2]
        x, y, rw, rh = self.cfg["roi"]
        x0, y0 = int(x * w), int(y * h)
        return x0, y0, max(8, int(rw * w)), max(8, int(rh * h))

    # -- un cuadro
    def paso(self, frame: np.ndarray, pista=None) -> np.ndarray:
        gris = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
        x, y, rw, rh = self._roi_px(frame.shape)
        ahora = time.monotonic()
        bbox = None
        frac = 0.0

        if self.estado == CALIBRANDO:
            if self._calentar > 0:
                self._calentar -= 1
            else:
                self._calib.append(gris)
            if len(self._calib) >= self.cfg["calib_frames"]:
                self.bg = np.median(np.stack(self._calib), axis=0)
                self._calib = []
                self.estado = VACIO
        else:
            offset = float(gris.mean() - self.bg.mean())
            d = np.abs(gris[y:y + rh, x:x + rw] - self.bg[y:y + rh, x:x + rw] - offset)
            mask = ((d > self.cfg["umbral_pix"]).astype(np.uint8)) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k15)
            frac = float(np.count_nonzero(mask)) / mask.size
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            sig = [c for c in cnts if cv2.contourArea(c) > 0.01 * mask.size]
            if sig:
                rects = [cv2.boundingRect(c) for c in sig]
                bx0 = min(r[0] for r in rects)
                by0 = min(r[1] for r in rects)
                bx1 = max(r[0] + r[2] for r in rects)
                by1 = max(r[1] + r[3] for r in rects)
                bbox = (bx0, by0, bx1, by1)
                if sum(cv2.contourArea(c) > 0.04 * mask.size for c in sig) >= 2:
                    self._dobles += 1
            self._transicion(frame, gris, frac, bbox, ahora, (x, y, rw, rh), pista)

        with self.lock:
            self.snap = {"estado": self.estado, "presencia": round(frac, 4), "fuente": self.fuente_nombre,
                         "modelo_listo": bool(self.clf.listo), "modelo_error": self.clf.error,
                         "en_vivo": None if self.en_vivo is None else {i: round(float(p), 4) for i, p in zip(self.ids, self.en_vivo)},
                         "cfg": dict(self.cfg)}
        return self._dibujar(frame, (x, y, rw, rh), bbox)

    def _transicion(self, frame, gris, frac, bbox, ahora, roi, pista):
        c = self.cfg
        e = self.estado
        if e == VACIO:
            if frac > c["entrada"]:
                self.estado, self._k = ENTRANDO, 1
            elif frac < c["salida"]:
                self.bg = self.bg * 0.97 + gris * 0.03  # sigue la luz de la cinta vacía
        elif e == ENTRANDO:
            if frac > c["entrada"]:
                self._k += 1
                if self._k >= c["k_entrada"]:
                    self._reiniciar_objeto()
                    self._t_ini = ahora
                    self._pista = pista
                    self.estado = PRESENTE
            else:
                self.estado, self._k = VACIO, 0
        elif e in (PRESENTE, SALIENDO):
            if frac > c["salida"]:
                self.estado = PRESENTE
                self._n_cuadros += 1
                self._acumular(frame, bbox, roi, ahora)
                if ahora - self._t_ini > c["max_presente_s"]:
                    self._emitir(ahora, extra="prolongado")
                    self.estado = RETENIDO
                    self._t_salida = ahora
            else:
                if e == PRESENTE:
                    self.estado, self._t_salida = SALIENDO, ahora
                elif ahora - self._t_salida >= c["sostener_salida_s"]:
                    self._emitir(ahora)
                    self.estado = VACIO
        elif e == RETENIDO:
            if frac > c["salida"]:
                self._t_salida = ahora
            elif ahora - self._t_salida >= c["sostener_salida_s"]:
                self.estado = VACIO

    def _acumular(self, frame, bbox, roi, ahora):
        if bbox is None or not self.clf.listo or ahora - self._t_clasif < self.cfg["clasificar_cada_s"]:
            return
        x, y, rw, rh = roi
        bx0, by0, bx1, by1 = bbox
        peso = 0.25 if (bx0 <= 2 or bx1 >= rw - 2) else 1.0  # objeto cortado por el borde: vale menos
        mx, my = int((bx1 - bx0) * 0.12), int((by1 - by0) * 0.12)
        H, W = frame.shape[:2]
        crop = frame[max(0, y + by0 - my):min(H, y + by1 + my), max(0, x + bx0 - mx):min(W, x + bx1 + mx)]
        if crop.size == 0:
            return
        if self.asinc:
            if self._pendientes:  # ya hay una lectura en curso: no se acumulan pendientes
                return
            self._t_clasif = ahora
            with self._cv:
                self._pendientes += 1
            self._cola.put((self._gen, peso, crop.copy()))
            return
        probs = self.clf.predecir(crop)
        self._t_clasif = ahora
        self._aplicar(peso, probs, crop)

    def _aplicar(self, peso, probs, crop):
        self._suma += peso * probs
        self._peso += peso
        self._n_clasif += 1
        self.en_vivo = self._suma / self._peso  # lo que decide el evento: promedio ponderado de todas las lecturas
        if peso * float(probs.max()) > self._mejor[0]:
            self._mejor = (peso * float(probs.max()), crop.copy())

    def _trabajador(self):
        """Clasifica en un hilo aparte para que el video no se congele mientras el modelo piensa."""
        while True:
            gen, peso, crop = self._cola.get()
            try:
                probs = self.clf.predecir(crop)
            except Exception:  # noqa: BLE001
                probs = None
            with self._cv:
                if probs is not None and gen == self._gen:  # si el objeto ya salió, la lectura tardía se descarta
                    self._aplicar(peso, probs, crop)
                self._pendientes -= 1
                self._cv.notify_all()

    def _emitir(self, ahora, extra=None):
        with self._cv:
            self._cv.wait_for(lambda: self._pendientes == 0, timeout=8)  # espera la lectura en curso
            self._emitir_ya(ahora, extra)

    def _emitir_ya(self, ahora, extra=None):
        if self._peso <= 0:  # nunca se pudo clasificar (modelo cargando o objeto muy fugaz)
            return
        probs = self._suma / self._peso
        i = int(np.argmax(probs))
        conf = float(probs[i])
        clase = self.ids[i]
        flags = [extra] if extra else []
        if conf < self.cfg["min_conf"]:
            clase = "indeterminado"
            flags.append("baja_confianza")
        if self._dobles >= max(3, 0.3 * self._n_cuadros):
            flags.append("posible_doble")
        alerta = int(clase != "indeterminado" and bool(self.clases[i].get("alerta")))
        ev = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "clase": clase, "conf": round(conf, 4),
              "n_cuadros": self._n_cuadros, "n_clasif": self._n_clasif, "duracion": round(ahora - self._t_ini, 2),
              "alerta": alerta, "flags": ",".join(flags), "fuente": self.fuente_nombre, "pista": self._pista,
              "probs": {k: round(float(p), 4) for k, p in zip(self.ids, probs)}}
        ev["id"] = self.reg.agregar(**ev)
        if self._mejor[1] is not None:
            nombre = f"{ev['id']}.jpg"
            cv2.imwrite(os.path.join(CROPS, nombre), self._mejor[1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.reg.poner_recorte(ev["id"], nombre)
            ev["recorte"] = nombre
        self.ultimo_evento = ev
        self._reiniciar_objeto()

    def _dibujar(self, frame, roi, bbox):
        out = frame.copy()
        x, y, rw, rh = roi
        col = COLOR[self.estado]
        cv2.rectangle(out, (x, y), (x + rw, y + rh), col, 2)
        if bbox:
            cv2.rectangle(out, (x + bbox[0], y + bbox[1]), (x + bbox[2], y + bbox[3]), (255, 255, 255), 1)
        txt = self.estado
        if self.en_vivo is not None and self.estado in (PRESENTE, SALIENDO):
            j = int(np.argmax(self.en_vivo))
            txt += f"  {self.ids[j]} {self.en_vivo[j]:.0%}"
        cv2.putText(out, txt, (x + 6, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        return out
