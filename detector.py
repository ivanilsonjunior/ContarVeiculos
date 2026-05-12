"""
detector.py — Thread de detecção de veículos
=============================================
Captura frames do player da webcam de Natal,
roda YOLO e mantém o histórico de contagens
para cálculo de média do último minuto e
média móvel dos últimos 10 minutos.
"""

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

# ─── Configurações ────────────────────────────────────────────────────────────

PLAYER_URL      = "https://www.vision-environnement.com/live/player/natal20.php"
PAGE_REFERER    = "https://www.vision-environnement.com/webcam/brasil/rio-grande-do-norte/3638-natal-monza-palace-hotel/"
CAPTURE_INTERVAL = 10          # segundos entre capturas
CONFIANCA_MINIMA  = 0.35

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Referer": PAGE_REFERER,
}

# ─── Estrutura de uma amostra ─────────────────────────────────────────────────

class Amostra:
    __slots__ = ("ts", "counts", "total", "frame_b64")

    def __init__(self, ts: datetime, counts: dict, frame_b64: str | None = None):
        self.ts       = ts
        self.counts   = counts                      # {"car":n, "motorcycle":n, ...}
        self.total    = sum(counts.values())
        self.frame_b64 = frame_b64                  # JPEG base64 do último frame

# ─── Detector (singleton thread) ──────────────────────────────────────────────

class VehicleDetector:
    """
    Roda em background. Expõe:
      - last_sample      → Amostra mais recente
      - avg_last_minute  → média do último minuto (float)
      - moving_avg_10min → média móvel dos últimos 10 minutos (float)
      - history_10min    → lista de (ts, avg_per_minute) dos últimos 10 min
      - all_samples      → lista completa de amostras (máx. 720 = 2h)
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self._lock          = threading.Lock()
        self._model         = None
        self._model_name    = model_name
        self._stream_url    = None

        # Amostras brutas — guardamos as últimas 2 horas (720 amostras × 10s)
        self._raw: deque[Amostra] = deque(maxlen=720)

        # Médias por minuto — guardamos os últimos 10 minutos
        self._min_avgs: deque[tuple[datetime, float]] = deque(maxlen=10)

        self.status         = "iniciando"
        self._thread        = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ── Propriedades públicas (thread-safe) ───────────────────────────────────

    @property
    def last_sample(self) -> Amostra | None:
        with self._lock:
            return self._raw[-1] if self._raw else None

    @property
    def avg_last_minute(self) -> float:
        """Média de veículos nas amostras do último minuto."""
        now = datetime.now(timezone.utc)
        with self._lock:
            recentes = [
                s.total for s in self._raw
                if (now - s.ts).total_seconds() <= 60
            ]
        return round(sum(recentes) / len(recentes), 2) if recentes else 0.0

    @property
    def moving_avg_10min(self) -> float:
        """Média móvel simples das últimas 10 médias-por-minuto."""
        with self._lock:
            vals = [v for _, v in self._min_avgs]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    @property
    def history_10min(self) -> list[dict]:
        """Histórico dos últimos 10 pontos (1 por minuto) para o gráfico."""
        with self._lock:
            return [
                {"ts": ts.isoformat(), "avg": round(avg, 2)}
                for ts, avg in self._min_avgs
            ]

    @property
    def all_samples(self) -> list[dict]:
        """Todas as amostras brutas (últimas 2h), do mais antigo ao mais recente."""
        with self._lock:
            return [
                {
                    "ts":    s.ts.isoformat(),
                    "total": s.total,
                    **s.counts,
                }
                for s in self._raw
            ]

    def snapshot_b64(self) -> str | None:
        """Último frame anotado em base64 JPEG (para o endpoint /snapshot)."""
        s = self.last_sample
        return s.frame_b64 if s else None

    # ── Loop interno ──────────────────────────────────────────────────────────

    def _loop(self):
        log.info("Carregando modelo YOLO…")
        self._model = YOLO(self._model_name)
        log.info("Modelo carregado.")

        # Tenta extrair stream HLS do player
        self._stream_url = self._extrair_stream()

        last_min_bucket = None   # controla quando avançar o bucket de minuto
        min_bucket_samples: list[float] = []

        while True:
            ts_inicio = time.monotonic()
            now_utc   = datetime.now(timezone.utc)

            # ── 1. Captura o frame ────────────────────────────────────────────
            frame = self._capturar()
            if frame is None:
                self.status = "sem_imagem"
                time.sleep(CAPTURE_INTERVAL)
                continue

            # ── 2. Detecta veículos ───────────────────────────────────────────
            counts, frame_anotado = self._detectar(frame)
            total = sum(counts.values())
            self.status = "ok"

            # ── 3. Codifica frame em base64 ───────────────────────────────────
            ok, buf = cv2.imencode(".jpg", frame_anotado, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = buf.tobytes().hex() if ok else None   # hex para serialização simples
            import base64
            b64_str = base64.b64encode(buf.tobytes()).decode() if ok else None

            # ── 4. Armazena amostra ───────────────────────────────────────────
            amostra = Amostra(ts=now_utc, counts=counts, frame_b64=b64_str)
            with self._lock:
                self._raw.append(amostra)

            # ── 5. Atualiza bucket de minuto ──────────────────────────────────
            minuto_atual = now_utc.replace(second=0, microsecond=0)
            min_bucket_samples.append(total)

            if last_min_bucket is None:
                last_min_bucket = minuto_atual

            if minuto_atual > last_min_bucket:
                avg = sum(min_bucket_samples) / len(min_bucket_samples)
                with self._lock:
                    self._min_avgs.append((last_min_bucket, avg))
                log.info(
                    f"Minuto {last_min_bucket.strftime('%H:%M')} → "
                    f"média {avg:.1f} veículos"
                )
                min_bucket_samples = [total]
                last_min_bucket = minuto_atual

            log.debug(f"[{now_utc.strftime('%H:%M:%S')}] total={total} {counts}")

            # ── 6. Aguarda até o próximo ciclo ────────────────────────────────
            elapsed = time.monotonic() - ts_inicio
            sleep_for = max(0, CAPTURE_INTERVAL - elapsed)
            time.sleep(sleep_for)

    # ── Captura ───────────────────────────────────────────────────────────────

    def _extrair_stream(self) -> str | None:
        """Tenta extrair URL HLS/RTSP do HTML do player."""
        try:
            r = requests.get(PLAYER_URL, headers=HEADERS, timeout=10)
            html = r.text
            padroes = [
                r"(https?://[^\s\"']+\.m3u8[^\s\"']*)",
                r"(rtsp://[^\s\"']+)",
                r"file\s*:\s*['\"]([^'\"]+)['\"]",
                r"source['\"]?\s*:\s*['\"]([^'\"]+)['\"]",
            ]
            for p in padroes:
                m = re.search(p, html, re.IGNORECASE)
                if m:
                    url = m.group(1)
                    log.info(f"Stream HLS encontrado: {url}")
                    return url
        except Exception as e:
            log.warning(f"Não foi possível extrair stream: {e}")

        # Fallback: imagem estática inferida a partir do player URL
        log.warning("Stream não encontrado — usando captura via OpenCV do player.")
        return None

    def _capturar(self) -> np.ndarray | None:
        """
        Tenta capturar um frame. Estratégia em cascata:
          1. OpenCV VideoCapture no PLAYER_URL (funciona se for HLS direto)
          2. requests + PIL na imagem JPEG estática do player
        """
        # Estratégia 1: OpenCV no stream/player
        if self._stream_url:
            cap = cv2.VideoCapture(self._stream_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return frame
            log.warning("Stream HLS falhou — tentando fallback JPEG.")
            self._stream_url = None   # desativa para não tentar mais

        # Estratégia 2: Imagem JPEG do player via requests
        # O player retorna um <img> ou redireciona para o JPEG
        try:
            r = requests.get(PLAYER_URL, headers=HEADERS, timeout=12, stream=True)
            ct = r.headers.get("Content-Type", "")
            if "image" in ct:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # O player é HTML — tenta extrair src de <img> ou src do player
            html = r.text
            # Procura por imagens JPEG/JPG referenciadas
            m = re.search(r'src=["\']([^"\']+\.(?:jpg|jpeg|JPG|JPEG))["\']', html)
            if m:
                img_url = m.group(1)
                if img_url.startswith("/"):
                    img_url = "https://www.vision-environnement.com" + img_url
                ri = requests.get(img_url, headers=HEADERS, timeout=10)
                ri.raise_for_status()
                img = Image.open(BytesIO(ri.content)).convert("RGB")
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        except Exception as e:
            log.error(f"Captura falhou: {e}")

        return None

    # ── Detecção ──────────────────────────────────────────────────────────────

    def _detectar(self, frame: np.ndarray) -> tuple[dict, np.ndarray]:
        """Roda YOLO e anota o frame. Retorna (counts, frame_anotado)."""
        results = self._model(frame, conf=CONFIANCA_MINIMA, verbose=False)[0]
        counts  = {nome: 0 for nome in VEHICLE_CLASSES.values()}

        CORES = {
            "car":        (0,   200, 255),
            "motorcycle": (0,   128, 255),
            "bus":        (220,  80,  80),
            "truck":      (80,  200,  80),
        }

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            nome  = VEHICLE_CLASSES[cls_id]
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            counts[nome] += 1
            cor = CORES[nome]
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame, f"{nome} {conf:.0%}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        # Painel HUD
        total = sum(counts.values())
        ts    = datetime.now().strftime("%H:%M:%S")
        linhas = [f"  {ts}", f"  TOTAL : {total}"] + \
                 [f"  {k:<11}: {v}" for k, v in counts.items()]
        ph = 14 + 20 * len(linhas)
        cv2.rectangle(frame, (8, 8), (230, ph), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (230, ph), (255, 255, 255), 1)
        for i, linha in enumerate(linhas):
            cor_txt = (0, 255, 150) if "TOTAL" in linha else (200, 200, 200)
            cv2.putText(frame, linha, (12, 26 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, cor_txt, 1)

        return counts, frame


# ── Instância global ──────────────────────────────────────────────────────────
_detector: VehicleDetector | None = None
_detector_lock = threading.Lock()

def get_detector(model_name: str = "yolov8n.pt") -> VehicleDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = VehicleDetector(model_name)
    return _detector
