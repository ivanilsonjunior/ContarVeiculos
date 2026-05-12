"""
app.py — Servidor Flask
=======================
Rotas:
  GET /                    → Dashboard HTML
  GET /api/stats           → JSON com todas as métricas
  GET /api/history         → JSON com histórico bruto (últimas 2h)
  GET /api/snapshot        → JPEG do último frame anotado
  GET /api/status          → Saúde do detector

Uso:
  python app.py
  python app.py --host 0.0.0.0 --port 5000 --model yolov8s.pt
"""

import argparse
import base64
import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, Response, request
from detector import get_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

app = Flask(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _detector():
    return get_detector(app.config.get("YOLO_MODEL", "yolov8n.pt"))


# ─── Rotas HTML ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── API ──────────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    """
    Retorna as métricas principais.

    Resposta JSON:
    {
      "status":           "ok" | "sem_imagem" | "iniciando",
      "timestamp":        "2026-05-11T22:00:00+00:00",
      "avg_last_minute":  3.4,          # média de veículos no último minuto
      "moving_avg_10min": 2.8,          # média móvel das últimas 10 médias-por-min
      "last_sample": {
        "ts":           "...",
        "total":        4,
        "car":          3,
        "motorcycle":   1,
        "bus":          0,
        "truck":        0
      },
      "history_10min": [
        {"ts": "...", "avg": 3.1},
        ...                             # até 10 pontos, 1 por minuto
      ]
    }
    """
    d    = _detector()
    last = d.last_sample

    payload = {
        "status":           d.status,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "avg_last_minute":  d.avg_last_minute,
        "moving_avg_10min": d.moving_avg_10min,
        "last_sample":      None,
        "history_10min":    d.history_10min,
    }

    if last:
        payload["last_sample"] = {
            "ts":    last.ts.isoformat(),
            "total": last.total,
            **last.counts,
        }

    return jsonify(payload)


@app.route("/api/history")
def api_history():
    """
    Retorna todas as amostras brutas das últimas 2 horas.
    Query param: ?limit=N (padrão: todas)

    Resposta JSON:
    {
      "count": 42,
      "samples": [
        {"ts": "...", "total": 4, "car": 3, "motorcycle": 1, "bus": 0, "truck": 0},
        ...
      ]
    }
    """
    d       = _detector()
    samples = d.all_samples
    limit   = request.args.get("limit", type=int)
    if limit:
        samples = samples[-limit:]
    return jsonify({"count": len(samples), "samples": samples})


@app.route("/api/snapshot")
def api_snapshot():
    """Retorna o último frame anotado como JPEG."""
    d   = _detector()
    b64 = d.snapshot_b64()
    if not b64:
        return Response("Sem imagem disponível.", status=503, mimetype="text/plain")
    img_bytes = base64.b64decode(b64)
    return Response(img_bytes, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/status")
def api_status():
    """Saúde rápida do serviço."""
    d = _detector()
    return jsonify({
        "status":        d.status,
        "samples_count": len(d.all_samples),
        "uptime_samples": len(d.all_samples),
    })


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natal Traffic Monitor")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  default=5000, type=int)
    parser.add_argument("--model", default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt",
                                 "yolov8m.pt", "yolov8l.pt"])
    args = parser.parse_args()

    app.config["YOLO_MODEL"] = args.model

    # Inicializa o detector antes de servir requisições
    get_detector(args.model)

    app.run(host=args.host, port=args.port, debug=False)
