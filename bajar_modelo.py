"""Descarga (y deja en caché) el modelo CLIP del MVP para poder trabajar sin internet después.

Usa open_clip, que pide solo el archivo de pesos necesario (no todo el repositorio de HF).
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import open_clip

open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
open_clip.get_tokenizer("ViT-B-32-quickgelu")
print("LISTO", flush=True)
