"""Local CLIP: rank storyboard tiles against a shot's subject on the CPU, with no judge call.

Free judge links are capped in requests per day, so the cheapest judge call is the one not made.
CLIP (ViT-B/32, the quantized ONNX export from Xenova/clip-vit-base-patch32, ~150 MB) runs on
`onnxruntime` without torch or a GPU. It is a *retriever*, not a judge: it orders candidates and
picks which tiles the judge gets to see. Whether a clip is usable is still decided by the judge,
local checks, frame-grid verify and the dedicated AI check.

Tile embeddings are cached next to the contact sheet, so a video that turns up in the searches of
many shots (common: one good "Kyoto 4K walking tour" serves a dozen) is embedded once.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np

from ..log import log

REPO = "Xenova/clip-vit-base-patch32"
FILES = {"vision": "onnx/vision_model_quantized.onnx", "text": "onnx/text_model_quantized.onnx",
         "tokenizer": "tokenizer.json"}
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIZE = 224
CONTEXT = 77


class Clip:
    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        self._lock = threading.Lock()
        self._vision = self._text = self._tok = None

    def _load(self):
        with self._lock:
            if self._vision is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            def get(f):
                try:                    # cached after the first run: no network round-trip
                    return hf_hub_download(REPO, f, cache_dir=str(self.dir), local_files_only=True)
                except Exception:
                    return hf_hub_download(REPO, f, cache_dir=str(self.dir))

            paths = {k: get(f) for k, f in FILES.items()}
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2          # shots run in parallel; keep the CPU for ffmpeg too
            prov = ["CPUExecutionProvider"]
            self._vision = ort.InferenceSession(paths["vision"], opts, providers=prov)
            self._text = ort.InferenceSession(paths["text"], opts, providers=prov)
            self._tok = Tokenizer.from_file(paths["tokenizer"])
            log.debug("CLIP loaded from %s", self.dir)

    @staticmethod
    def _pixels(img) -> np.ndarray:
        """CLIP preprocessing, except the whole 16:9 frame is kept (squashed to a square) rather
        than centre-cropped: a crop drops ~45% of a wide thumbnail, often the subject itself."""
        im = img.convert("RGB").resize((SIZE, SIZE), resample=3)        # bicubic
        a = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
        return a.transpose(2, 0, 1)

    def images(self, imgs: list) -> np.ndarray:
        """L2-normalised embeddings, shape (n, 512)."""
        self._load()
        out = []
        for i in range(0, len(imgs), 32):
            batch = np.stack([self._pixels(im) for im in imgs[i:i + 32]])
            out.append(self._vision.run(None, {"pixel_values": batch})[0])
        e = np.concatenate(out) if out else np.zeros((0, 512), np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)

    def text(self, s: str) -> np.ndarray:
        """L2-normalised embedding of `s`, shape (512,). One text per run: the export has no
        attention mask, and CLIP pools at the end-of-text token, so padding must not be added."""
        self._load()
        ids = self._tok.encode(s).ids[:CONTEXT]
        if ids[-1] != 49407:                     # keep the end-of-text token the model pools on
            ids[-1] = 49407
        e = self._text.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0][0]
        return e / max(float(np.linalg.norm(e)), 1e-8)


def sheet_tiles(jpeg: bytes, n: int, cols: int) -> list:
    """Cut a numbered contact sheet (see sources/storyboard.py) back into its `n` tiles."""
    from PIL import Image

    sheet = Image.open(io.BytesIO(jpeg)).convert("RGB")
    rows = -(-n // cols)
    w, h = sheet.width // cols, sheet.height // rows
    tiles = []
    for i in range(n):
        x, y = (i % cols) * w, (i // cols) * h
        # skip the number label in the top-left corner: it is not part of the video
        tiles.append(sheet.crop((x, y + h // 5, x + w, y + h)))
    return tiles


class TileIndex:
    """Per-video tile embeddings, cached on disk beside the storyboard."""

    def __init__(self, clip: Clip, cache_dir: Path):
        self.clip, self.dir = clip, Path(cache_dir) / "storyboards"
        self._mem: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def tiles(self, sb) -> np.ndarray:
        from ..sources.storyboard import SHEET_COLS

        with self._lock:
            if sb.video_id in self._mem:
                return self._mem[sb.video_id]
        f = self.dir / f"{sb.video_id}.clip.npy"
        e = None
        if f.exists():
            try:
                e = np.load(f)
            except (OSError, ValueError):
                e = None
        if e is None or len(e) != len(sb.times):
            e = self.clip.images(sheet_tiles(sb.jpeg, len(sb.times), SHEET_COLS))
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(f, e.astype(np.float16))
        e = e.astype(np.float32)
        with self._lock:
            self._mem[sb.video_id] = e
        return e

    def scores(self, sb, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of every tile of `sb` to the text embedding `query`."""
        return self.tiles(sb) @ query


def query_text(brief) -> str:
    """What CLIP should look for: the required subject, plus the names involved."""
    names = [e for e in brief.entities if e.lower() not in (brief.subject or "").lower()][:2]
    core = brief.subject or brief.visual
    return f"a photo of {core}" + (f", {', '.join(names)}" if names else "")
