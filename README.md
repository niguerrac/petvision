# PetVision — clasificación y conteo de envases en una cinta transportadora

Programa en Python con panel web (HTTPS) que mira una cinta transportadora con una cámara, **clasifica los envases plásticos
de líquidos** (botellas, bidones, envases de limpieza) que pasan de a uno, **cuenta cada objeto una sola vez** y guarda un
registro para analizarlo después. Avisa si se cuela un envase de **vidrio**.

Pensado para correr en una **Raspberry Pi 4** con una **Pi Camera**, y accesible desde cualquier notebook del mismo WiFi.

> **Estado del proyecto: MVP.** El clasificador es CLIP *zero-shot* (no está entrenado con fotos de la cinta real).
> La instalación en la Raspberry Pi y la Pi Camera **todavía no se han probado en el hardware real** (el programa se desarrolló y probó
> en un PC con Linux y una webcam USB). Ver [Estado de las pruebas](#estado-de-las-pruebas).

## Qué hace

- Panel con video en vivo, zona de detección ajustable, conteo por clase, objeto en curso con sus probabilidades, último evento con su foto,
  tabla de eventos y gráfico por minuto.
- Registro de cada objeto en SQLite (`data/eventos.db`) y descarga en CSV.
- Fuentes de video: **Pi Camera (CSI)**, **webcam USB** y **simulación** con fotos (para probar sin cámara).
- Clases: botella plástica, bidón plástico, envase de limpieza, vidrio (alerta), otro objeto, indeterminado (confianza baja).
  Se editan en [`clases.json`](clases.json).

## Hardware usado

| Pieza | Detalle |
|---|---|
| Computador | **Raspberry Pi 4 Model B, 8 GB de RAM** |
| Cámara | **Pi Camera** (módulo CSI) conectada **directo a la placa por cable plano**. El modelo exacto (v1, v2, v3, HQ) no está verificado: cualquier módulo compatible con `libcamera`/`picamera2` debería servir |
| Almacenamiento | tarjeta **microSD de 32 GB** (vacía, sin sistema) |
| Alimentación | fuente USB-C para Pi 4 (la oficial es de 5 V / 3 A) |
| Refrigeración | disipadores adhesivos como mínimo: el clasificador mantiene la CPU ocupada. Un ventilador de Pi 5 **no** se conecta a una Pi 4 (no tiene ese conector) |
| Red | WiFi o cable de red, y un notebook en la misma red para abrir el panel |

## Instalación en la Raspberry Pi 4 desde una tarjeta SD vacía

Tiempo aproximado: 30–60 minutos, casi todo esperando descargas.

### 1. Grabar el sistema operativo en la SD (desde tu notebook)

1. Instala **Raspberry Pi Imager**: <https://www.raspberrypi.com/software/>.
2. En el Imager elige:
   - **Dispositivo**: Raspberry Pi 4
   - **Sistema operativo**: *Raspberry Pi OS (other)* → **Raspberry Pi OS Lite (64-bit)**.
     Tiene que ser de **64 bits**: PyTorch no se instala en la versión de 32 bits.
   - **Almacenamiento**: la microSD de 32 GB.
3. Pulsa **Siguiente** y, cuando pregunte por la personalización, elige **Editar configuración** y completa:
   - **Nombre de host**: `petvision`
   - **Usuario y contraseña** (anótalos)
   - **WiFi**: nombre y clave de tu red; país **CL**
   - **Zona horaria**: `America/Santiago`
   - Pestaña *Servicios*: **Habilitar SSH** (autenticación por contraseña)
4. Graba la tarjeta (borra todo lo que tuviera).

### 2. Armar el hardware (con la Pi apagada)

1. Conecta la **Pi Camera** al conector CSI de la placa con el cable plano: levanta la pestaña del conector, inserta el cable hasta el fondo
   con los contactos hacia el lado correcto, y baja la pestaña. **Hazlo con la Pi sin corriente.**
2. Pega los disipadores, inserta la microSD y conecta la fuente. La primera vez tarda 1–2 minutos en arrancar.

### 3. Conectarte por SSH (desde tu notebook)

```sh
ssh TU_USUARIO@petvision.local
```

Si `petvision.local` no responde, usa la IP que muestra tu router: `ssh TU_USUARIO@192.168.x.x`.

### 4. Descargar e instalar PetVision (dentro de la Pi)

```sh
sudo apt update && sudo apt install -y git
git clone https://github.com/niguerrac/petvision.git
cd petvision
./instalar_pi.sh
```

El instalador (tarda varios minutos, sobre todo PyTorch y el modelo):

1. instala los paquetes del sistema (OpenCV, numpy, `picamera2`, OpenSSL);
2. crea el entorno Python e instala FastAPI, PyTorch (CPU), OpenCLIP, etc.;
3. descarga el modelo CLIP (~600 MB) a la caché de Hugging Face;
4. genera un certificado HTTPS autofirmado;
5. crea y activa el servicio `petvision` (arranca solo cada vez que enciendes la Pi).

> **Modelo sin descargar en la Pi** (si la red es lenta): desde otro equipo que ya lo tenga en `~/.cache/huggingface/hub/`:
> ```sh
> ssh TU_USUARIO@petvision.local "mkdir -p ~/.cache/huggingface/hub"
> scp -r ~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai TU_USUARIO@petvision.local:.cache/huggingface/hub/
> ```
> Hazlo **antes** de `./instalar_pi.sh` y el instalador lo detecta.

### 5. Abrir el panel

En el navegador de tu notebook: **`https://petvision.local:8443/`** (o `https://IP_DE_LA_PI:8443/`).

El navegador mostrará un aviso porque el certificado es autofirmado: *Avanzado* → *Continuar*. Es normal y solo ocurre la primera vez por equipo.

### 6. Primer uso

1. Arriba a la izquierda debe verse la fuente **Pi Camera (CSI)** y el estado **VACÍO** (tras unos 5 segundos: la cámara espera a ajustar la exposición antes de aprender el fondo).
2. Con la cinta **vacía** pulsa **Calibrar fondo** (la cámara aprende cómo se ve la cinta sin objetos).
3. Ajusta con los deslizadores la **zona** (el recuadro) para que cubra por donde pasan los objetos.
4. Pasa un envase por la cinta: el estado debe ir `VACÍO → ENTRANDO → PRESENTE → SALIENDO → VACÍO` y sumar **un** evento.
5. Antes de una medición real, pulsa **Reiniciar conteo** (guarda un respaldo de la base antes de vaciarla).

### Si algo falla

| Síntoma | Qué revisar |
|---|---|
| `./instalar_pi.sh` dice que el sistema no es de 64 bits | Grabar de nuevo la SD con *Raspberry Pi OS Lite (64-bit)* |
| El panel muestra un error de fuente / no hay imagen | `rpicam-hello --list-cameras` debe listar la cámara. Si no aparece, apaga la Pi y revisa el cable plano y el sentido del conector |
| "Error del modelo" en la esquina superior derecha | El modelo no está en caché o no se pudo descargar: ver el recuadro de modelo arriba y reiniciar con `sudo systemctl restart petvision` |
| No abre `https://petvision.local:8443/` | `sudo systemctl status petvision` y `journalctl -u petvision -f`; probar con la IP en lugar del nombre |
| Quiero detenerlo / reiniciarlo | `sudo systemctl stop petvision` / `sudo systemctl restart petvision` |

## Cómo evita contar dos veces el mismo objeto

Máquina de estados sobre una zona fija (ROI): `VACÍO → ENTRANDO → PRESENTE → SALIENDO → evento único`.

- **Presencia** = diferencia contra el fondo de la cinta vacía, con histéresis (umbral alto para entrar y bajo para salir).
- Mientras el objeto está en la zona se clasifica cada ~0,3 s y se **promedian** las probabilidades
  (las lecturas con el objeto cortado por el borde pesan menos).
- El evento se registra **al salir**, tras 1 s sin presencia. Si el objeto reaparece antes, es el mismo objeto.
- Un objeto quieto, o la cinta detenida, **no** vuelve a contarse. Si se queda más de 25 s se registra una vez con la nota `prolongado`.
- La clasificación corre en un hilo aparte, así que un modelo lento no congela el video ni pierde objetos.
- Notas posibles en cada evento: `baja_confianza` (queda como `indeterminado`), `posible_doble` (dos objetos juntos), `prolongado`.

## Datos

- `data/eventos.db` (SQLite): fecha, clase, confianza, cuántas lecturas, duración, alerta de vidrio, notas, foto y probabilidades.
- `data/crops/<id>.jpg`: la mejor foto de cada objeto (sirve para revisar errores y para reentrenar).
- Panel: botón **Descargar CSV**. API: `/api/estado`, `/api/eventos`, `/api/estadisticas`, `/api/eventos.csv`.

## Probarlo en un PC (sin Raspberry Pi)

```sh
git clone https://github.com/niguerrac/petvision.git && cd petvision
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install fastapi "uvicorn[standard]" open_clip_torch pillow numpy opencv-python-headless
./certs.sh                              # certificado HTTPS autofirmado
PETVISION_FUENTE=webcam ./run.sh        # o PETVISION_FUENTE=simulacion
```

Abrir `https://localhost:8443/`. La primera vez descarga el modelo CLIP (~600 MB).

**Modo simulación:** pon fotos de botellas en `simulacion/` (o apunta `PETVISION_SIM` a otra carpeta). Cruzan una cinta virtual de izquierda
a derecha y sirven para ver el conteo sin cámara. Prueba automática: `.venv/bin/python prueba_conteo.py 24 clip` pasa 24 objetos simulados y
comprueba que se cuenten exactamente 24.

## Configuración (variables de entorno)

| Variable | Qué hace | Por defecto |
|---|---|---|
| `PETVISION_FUENTE` | `webcam`, `picamera` o `simulacion` | `webcam` (en la Pi, el servicio usa `picamera`) |
| `PETVISION_TOKEN` | Clave de acceso: la dirección pasa a `https://…:8443/?t=clave` | sin clave |
| `PETVISION_FPS` | Cuadros procesados por segundo | `8` (el servicio de la Pi usa `6`) |
| `PETVISION_HILOS` | Hilos de PyTorch | mitad de los núcleos (la Pi usa `3`) |
| `PETVISION_ASYNC` | Clasificar en un hilo aparte (`1`/`0`) | `1` |
| `PETVISION_SIM` / `PETVISION_SIM_CLASES` | Carpeta de fotos de la simulación y filtro por subcarpeta | `simulacion/` / todas |
| `PETVISION_DATA` | Carpeta de datos (base de eventos y fotos) | `data/` |
| `PORT` | Puerto HTTPS | `8443` |

En la Pi se cambian en `/etc/systemd/system/petvision.service`; luego `sudo systemctl daemon-reload && sudo systemctl restart petvision`.

## Estado de las pruebas

**Probado** (PC con Linux, webcam USB y simulación):
- El conteo sin dobles: 24 objetos simulados → 24 eventos, con CLIP real y con un clasificador lento de 1,5 s por lectura.
- El panel por HTTPS, la API, la webcam, la alerta de vidrio y la descarga de datos.

**No probado todavía:**
- El instalador y el servicio en una Raspberry Pi real, y la fuente de la **Pi Camera** (`picamera2`).
- La **velocidad** de CLIP en una Pi 4: estimo del orden de 1–2 s por lectura, pero no está medida. Cada evento guarda cuántas lecturas hizo y cuánto duró.
- La precisión con envases reales en una cinta real. CLIP *zero-shot* distingue bien vidrio de una botella plástica de agua, pero es irregular con envases
  de limpieza. **Es solo para el MVP.**

## Hoja de ruta

1. Tomar fotos en la cinta real (≈150–200 por clase, más cinta vacía) con la propia cámara y luz.
2. Entrenar un clasificador liviano (MobileNetV2, *transfer learning*) y evaluarlo con objetos distintos de los de entrenamiento.
3. Exportarlo a ONNX int8 y reemplazar CLIP (misma interfaz `predecir(recorte)`), medido en la Pi.
4. Prueba larga de una hora comparando con conteo manual, y análisis estadístico de los registros.

## Notas sobre datos y modelos

- Usa [OpenCLIP](https://github.com/mlfoundations/open_clip) con los pesos `openai` del modelo `ViT-B-32-quickgelu`.
- El repositorio **no incluye** fotos ni datos de pruebas.
- Aún no se ha definido una licencia para este código.
