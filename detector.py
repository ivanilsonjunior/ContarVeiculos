"""
detector.py — Multi-stream vehicle detection
"""

import base64
import re
import time
import threading
import logging
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

log = logging.getLogger("detector")

CAPTURE_INTERVAL = 3
CONFIANCA_MINIMA  = 0.25
STREAM_TTL        = 25 * 60

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

CORES = {
    "car":        (0,   200, 255),
    "motorcycle": (0,   128, 255),
    "bus":        (220,  80,  80),
    "truck":      (80,  200,  80),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
}

# ── Shared YOLO model ─────────────────────────────────────────────────────────

_model_lock = threading.Lock()
_yolo: YOLO | None = None
_yolo_name = "yolov8m.pt"


def load_model(name: str = "yolov8m.pt") -> YOLO:
    global _yolo, _yolo_name
    with _model_lock:
        if _yolo is None:
            log.info(f"Carregando modelo YOLO: {name}")
            _yolo = YOLO(name)
            _yolo_name = name
            log.info("Modelo carregado.")
    return _yolo


# ── ROI ───────────────────────────────────────────────────────────────────────

def _in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray-casting point-in-polygon (normalized coords)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ── Sample ────────────────────────────────────────────────────────────────────

class Amostra:
    __slots__ = ("ts", "counts", "total", "frame_b64")

    def __init__(self, ts: datetime, counts: dict, frame_b64: str | None = None):
        self.ts        = ts
        self.counts    = counts
        self.total     = sum(counts.values())
        self.frame_b64 = frame_b64


# ── Per-stream detector ───────────────────────────────────────────────────────

class StreamDetector:
    """One background thread per configured stream."""

    def __init__(
        self,
        stream_id: str,
        name: str,
        url: str,
        detection_zone: list | None,
        on_sample=None,          # callback(stream_id, Amostra)
    ):
        self.stream_id = stream_id
        self._lock     = threading.RLock()
        self._stop_evt = threading.Event()

        self._name  = name
        self._url   = url
        self._zone  = detection_zone   # [[x_norm, y_norm], ...] or None

        self._on_sample       = on_sample
        self._stream_url      = None
        self._stream_at       = 0.0

        self._raw:      deque[Amostra]         = deque(maxlen=28800)
        self._min_avgs: deque[tuple]           = deque(maxlen=28800)

        self.status = "iniciando"

        self._thread = threading.Thread(
            target=self._loop, name=f"det-{stream_id[:8]}", daemon=True
        )
        self._thread.start()

    # ── Config updates (hot-reload) ───────────────────────────────────────────

    def update_config(self, name=None, url=None, detection_zone=None):
        with self._lock:
            if name is not None:
                self._name = name
            if url is not None:
                self._url = url
                self._stream_url = None
            if detection_zone is not None:
                self._zone = detection_zone

    def clear_zone(self):
        with self._lock:
            self._zone = None

    def stop(self):
        self._stop_evt.set()

    # ── Seed from DB on startup ───────────────────────────────────────────────

    def seed(self, rows: list[dict]):
        bucket_map: dict = {}
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            counts = {k: r[k] for k in ("car", "motorcycle", "bus", "truck")}
            self._raw.append(Amostra(ts=ts, counts=counts))

            bk = int(ts.timestamp() // 3) * 3
            bk_dt = datetime.fromtimestamp(bk, tz=timezone.utc)
            bucket_map.setdefault(bk_dt, []).append(r["total"])

        for bk_dt in sorted(bucket_map):
            vals = bucket_map[bk_dt]
            self._min_avgs.append((bk_dt, sum(vals) / len(vals)))

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_sample(self) -> Amostra | None:
        with self._lock:
            return self._raw[-1] if self._raw else None

    @property
    def avg_last_minute(self) -> float:
        from datetime import timezone as tz
        now = datetime.now(tz.utc)
        with self._lock:
            vals = [s.total for s in self._raw if (now - s.ts).total_seconds() <= 60]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    @property
    def moving_avg_10min(self) -> float:
        with self._lock:
            vals = [v for _, v in list(self._min_avgs)[-200:]]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    @property
    def history_10min(self) -> list[dict]:
        with self._lock:
            items = list(self._min_avgs)[-200:]
        return [{"ts": ts.isoformat(), "avg": round(avg, 2)} for ts, avg in items]

    @property
    def history_24h(self) -> list[dict]:
        with self._lock:
            return [{"ts": ts.isoformat(), "avg": round(avg, 2)} for ts, avg in self._min_avgs]

    @property
    def all_samples(self) -> list[dict]:
        with self._lock:
            return [{"ts": s.ts.isoformat(), "total": s.total, **s.counts} for s in self._raw]

    def snapshot_b64(self) -> str | None:
        s = self.last_sample
        return s.frame_b64 if s else None

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self):
        model = load_model(_yolo_name)
        with self._lock:
            url = self._url
        self._stream_url = self._extract_stream(url)
        self._stream_at  = time.monotonic()

        last_bucket: datetime | None = None
        bucket_vals: list[float]     = []

        while not self._stop_evt.is_set():
            t0      = time.monotonic()
            now_utc = datetime.now(timezone.utc)

            with self._lock:
                url  = self._url
                zone = self._zone

            frame = self._capture(url)
            if frame is None:
                self.status = "sem_imagem"
                time.sleep(CAPTURE_INTERVAL)
                continue

            counts, frame_ann = self._detect(model, frame, zone)
            total = sum(counts.values())
            self.status = "ok"

            ok, buf = cv2.imencode(".jpg", frame_ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64_str = base64.b64encode(buf.tobytes()).decode() if ok else None

            amostra = Amostra(ts=now_utc, counts=counts, frame_b64=b64_str)
            with self._lock:
                self._raw.append(amostra)

            if self._on_sample:
                self._on_sample(self.stream_id, amostra)

            # 3-second bucket
            bk_ts = int(now_utc.timestamp() // 3) * 3
            bk_dt = datetime.fromtimestamp(bk_ts, tz=timezone.utc)
            bucket_vals.append(total)

            if last_bucket is None:
                last_bucket = bk_dt

            if bk_dt > last_bucket:
                avg = sum(bucket_vals) / len(bucket_vals)
                with self._lock:
                    self._min_avgs.append((last_bucket, avg))
                log.info(
                    f"[{self._name}] bucket {last_bucket.strftime('%H:%M:%S')} "
                    f"→ avg {avg:.1f}"
                )
                bucket_vals = [total]
                last_bucket = bk_dt

            time.sleep(max(0, CAPTURE_INTERVAL - (time.monotonic() - t0)))

    # ── Capture ───────────────────────────────────────────────────────────────

    # YouTube video ID from any common URL form
    _YT_ID = re.compile(
        r'(?:youtube\.com/(?:watch\?(?:.*&)?v=|live/|shorts/|embed/)'
        r'|youtu\.be/)'
        r'([A-Za-z0-9_-]{11})',
        re.IGNORECASE,
    )

    def _extract_stream(self, page_url: str) -> str | None:
        # Direct stream URLs — use as-is
        low = page_url.lower()
        if any(low.endswith(x) or (x + "?") in low for x in (".m3u8", ".ts")):
            return page_url
        if low.startswith(("rtsp://", "rtmp://")):
            return page_url

        # YouTube URL supplied directly by the user
        yt_direct = self._YT_ID.search(page_url)
        if yt_direct:
            log.info(f"[{self.stream_id}] YouTube URL detectado direto: {yt_direct.group(1)}")
            return self._extract_youtube(yt_direct.group(1))

        # Generic web player page — scrape for stream/embed references
        try:
            r    = requests.get(page_url, headers={**HEADERS, "Referer": page_url}, timeout=10)
            html = r.text
            for pat in [
                r"(https?://[^\s\"']+\.m3u8[^\s\"']*)",
                r"(rtsp://[^\s\"']+)",
                r'file\s*:\s*[\'"]([^\'"]+)[\'"]',
                r'source[\'"]?\s*:\s*[\'"]([^\'"]+)[\'"]',
            ]:
                m = re.search(pat, html, re.IGNORECASE)
                if m:
                    return m.group(1)

            yt_embed = self._YT_ID.search(html)
            if yt_embed:
                log.info(f"[{self.stream_id}] YouTube embed encontrado na página: {yt_embed.group(1)}")
                return self._extract_youtube(yt_embed.group(1))
        except Exception as e:
            log.warning(f"[{self.stream_id}] extração falhou: {e}")
        return None

    def _extract_youtube(self, vid_id: str) -> str | None:
        try:
            import yt_dlp
        except ImportError:
            log.error("yt-dlp não instalado. Execute: pip install yt-dlp")
            return None
        try:
            opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio/best",
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={vid_id}", download=False
                )
                url = info.get("url")
                if not url and info.get("formats"):
                    url = info["formats"][-1].get("url")
                if url:
                    log.info(f"[{self.stream_id}] URL YouTube obtida via yt-dlp ({vid_id})")
                else:
                    log.warning(f"[{self.stream_id}] yt-dlp não retornou URL para {vid_id}")
                return url
        except Exception as e:
            log.error(f"[{self.stream_id}] yt-dlp falhou para {vid_id}: {e}")
        return None

    def _capture(self, page_url: str) -> np.ndarray | None:
        # Renew stream URL if TTL expired
        if self._stream_url and (time.monotonic() - self._stream_at) > STREAM_TTL:
            nova = self._extract_stream(page_url)
            if nova:
                self._stream_url = nova
                self._stream_at  = time.monotonic()

        # Strategy 1: OpenCV VideoCapture
        if self._stream_url:
            cap = cv2.VideoCapture(self._stream_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return frame
            log.warning(f"[{self.stream_id}] stream falhou, renovando URL…")
            nova = self._extract_stream(page_url)
            if nova:
                self._stream_url = nova
                self._stream_at  = time.monotonic()
                cap2 = cv2.VideoCapture(nova)
                if cap2.isOpened():
                    ret, frame = cap2.read()
                    cap2.release()
                    if ret and frame is not None:
                        return frame
            self._stream_url = None

        # Strategy 2: static JPEG fallback
        try:
            hdrs = {**HEADERS, "Referer": page_url}
            r    = requests.get(page_url, headers=hdrs, timeout=12, stream=True)
            ct   = r.headers.get("Content-Type", "")
            if "image" in ct:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            html = r.text
            m = re.search(r'src=["\']([^"\']+\.(?:jpg|jpeg|JPG|JPEG))["\']', html)
            if m:
                img_url = m.group(1)
                if img_url.startswith("/"):
                    from urllib.parse import urlparse
                    p = urlparse(page_url)
                    img_url = f"{p.scheme}://{p.netloc}{img_url}"
                ri  = requests.get(img_url, headers=hdrs, timeout=10)
                ri.raise_for_status()
                img = Image.open(BytesIO(ri.content)).convert("RGB")
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            log.error(f"[{self.stream_id}] captura falhou: {e}")
        return None

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, model: YOLO, frame: np.ndarray, zone: list | None) -> tuple[dict, np.ndarray]:
        with _model_lock:
            results = model(frame, conf=CONFIANCA_MINIMA, verbose=False)[0]

        counts = {nome: 0 for nome in VEHICLE_CLASSES.values()}
        h, w   = frame.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            nome = VEHICLE_CLASSES[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if zone and len(zone) >= 3:
                cx_n = ((x1 + x2) / 2) / w
                cy_n = ((y1 + y2) / 2) / h
                if not _in_polygon(cx_n, cy_n, zone):
                    continue

            counts[nome] += 1
            cor = CORES[nome]
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(
                frame, f"{nome} {conf:.0%}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2,
            )

        # Draw ROI overlay
        if zone and len(zone) >= 3:
            pts     = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 212, 255))
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 212, 255), 2)

        # HUD
        with self._lock:
            name = self._name
        total  = sum(counts.values())
        ts_str = datetime.now().strftime("%H:%M:%S")
        linhas = [f"  {ts_str}", f"  {name[:16]}", f"  TOTAL : {total}"] + \
                 [f"  {k:<11}: {v}" for k, v in counts.items()]
        ph = 14 + 20 * len(linhas)
        cv2.rectangle(frame, (8, 8), (240, ph), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (240, ph), (255, 255, 255), 1)
        for i, linha in enumerate(linhas):
            cor_txt = (0, 255, 150) if "TOTAL" in linha else (200, 200, 200)
            cv2.putText(frame, linha, (12, 26 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, cor_txt, 1)

        return counts, frame


# ── Manager ───────────────────────────────────────────────────────────────────

class DetectorManager:
    def __init__(self):
        self._lock  = threading.Lock()
        self._dets: dict[str, StreamDetector] = {}

    def start(self, stream_id: str, name: str, url: str,
              detection_zone: list | None, on_sample=None) -> StreamDetector:
        with self._lock:
            if stream_id not in self._dets:
                self._dets[stream_id] = StreamDetector(
                    stream_id, name, url, detection_zone, on_sample
                )
        return self._dets[stream_id]

    def stop(self, stream_id: str):
        with self._lock:
            det = self._dets.pop(stream_id, None)
        if det:
            det.stop()

    def get(self, stream_id: str) -> StreamDetector | None:
        return self._dets.get(stream_id)

    def all(self) -> list[StreamDetector]:
        return list(self._dets.values())

    def update(self, stream_id: str, **kwargs):
        det = self.get(stream_id)
        if det:
            det.update_config(**kwargs)


_manager: DetectorManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> DetectorManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = DetectorManager()
    return _manager
