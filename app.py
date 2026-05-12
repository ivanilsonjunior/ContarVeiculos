"""
app.py — Servidor Flask multi-stream
=====================================
Rotas:
  GET  /                              → Dashboard HTML
  GET  /api/streams                   → Lista todos os fluxos
  POST /api/streams                   → Cria novo fluxo
  PUT  /api/streams/<id>              → Atualiza fluxo (nome, url, detection_zone)
  DELETE /api/streams/<id>            → Remove fluxo
  GET  /api/streams/<id>/stats        → Métricas do fluxo
  GET  /api/streams/<id>/snapshot     → JPEG do último frame anotado
  GET  /api/streams/<id>/status       → Saúde do fluxo

Uso:
  python app.py
  python app.py --host 0.0.0.0 --port 5000 --model yolov8s.pt
"""

import argparse
import base64
import logging
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, Response, request

import db
from detector import get_manager, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("app")
app = Flask(__name__)

DEFAULT_STREAM = {
    "id":   "natal-monza",
    "name": "Natal — Monza Palace",
    "url":  "https://www.vision-environnement.com/live/player/natal20.php",
}


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _on_sample(stream_id: str, amostra):
    """Persiste cada amostra no banco."""
    db.insert_sample(stream_id, amostra.ts, amostra.total, amostra.counts)


def _start_stream(s: dict):
    mgr = get_manager()
    det = mgr.start(
        stream_id      = s["id"],
        name           = s["name"],
        url            = s["url"],
        detection_zone = s.get("detection_zone"),
        on_sample      = _on_sample,
    )
    rows = db.load_history(s["id"])
    if rows:
        det.seed(rows)
        log.info(f"[{s['name']}] {len(rows)} amostras carregadas do banco.")
    return det


def bootstrap(model_name: str):
    db.init_db()
    db.prune_history()
    load_model(model_name)

    streams = db.list_streams()
    if not streams:
        log.info("Nenhum fluxo cadastrado — criando fluxo padrão.")
        db.create_stream(DEFAULT_STREAM["id"], DEFAULT_STREAM["name"], DEFAULT_STREAM["url"])
        streams = db.list_streams()

    for s in streams:
        if s.get("active", 1):
            _start_stream(s)


# ── HTML ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Stream CRUD ───────────────────────────────────────────────────────────────

@app.route("/api/streams", methods=["GET"])
def api_list_streams():
    streams = db.list_streams()
    mgr     = get_manager()
    for s in streams:
        det = mgr.get(s["id"])
        s["status"] = det.status if det else "parado"
    return jsonify(streams)


@app.route("/api/streams", methods=["POST"])
def api_create_stream():
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    url  = (body.get("url")  or "").strip()
    if not name or not url:
        return jsonify({"error": "name e url são obrigatórios"}), 400

    sid = str(uuid.uuid4())[:8]
    s   = db.create_stream(sid, name, url)
    _start_stream(s)
    return jsonify(s), 201


@app.route("/api/streams/<sid>", methods=["PUT"])
def api_update_stream(sid: str):
    s = db.get_stream(sid)
    if not s:
        return jsonify({"error": "não encontrado"}), 404

    body = request.get_json(force=True) or {}
    kwargs: dict = {}
    if "name"           in body: kwargs["name"]           = body["name"]
    if "url"            in body: kwargs["url"]            = body["url"]
    if "detection_zone" in body: kwargs["detection_zone"] = body["detection_zone"]

    updated = db.update_stream(sid, **kwargs)
    get_manager().update(sid, **kwargs)
    return jsonify(updated)


@app.route("/api/streams/<sid>", methods=["DELETE"])
def api_delete_stream(sid: str):
    s = db.get_stream(sid)
    if not s:
        return jsonify({"error": "não encontrado"}), 404
    get_manager().stop(sid)
    db.delete_stream(sid)
    return jsonify({"ok": True})


# ── Per-stream data ───────────────────────────────────────────────────────────

@app.route("/api/streams/<sid>/stats")
def api_stream_stats(sid: str):
    det = get_manager().get(sid)
    if not det:
        return jsonify({"error": "fluxo não encontrado ou parado"}), 404

    last = det.last_sample
    payload = {
        "status":           det.status,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "avg_last_minute":  det.avg_last_minute,
        "moving_avg_10min": det.moving_avg_10min,
        "last_sample":      None,
        "history_10min":    det.history_10min,
        "history_24h":      det.history_24h,
        "samples_count":    len(det.all_samples),
    }
    if last:
        payload["last_sample"] = {
            "ts":    last.ts.isoformat(),
            "total": last.total,
            **last.counts,
        }
    return jsonify(payload)


@app.route("/api/streams/<sid>/snapshot")
def api_stream_snapshot(sid: str):
    det = get_manager().get(sid)
    if not det:
        return Response("Fluxo não encontrado.", status=404, mimetype="text/plain")
    b64 = det.snapshot_b64()
    if not b64:
        return Response("Sem imagem disponível.", status=503, mimetype="text/plain")
    img_bytes = base64.b64decode(b64)
    return Response(img_bytes, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/streams/<sid>/status")
def api_stream_status(sid: str):
    det = get_manager().get(sid)
    if not det:
        return jsonify({"status": "parado", "samples_count": 0})
    return jsonify({"status": det.status, "samples_count": len(det.all_samples)})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natal Traffic Monitor")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  default=5000, type=int)
    parser.add_argument("--model", default="yolov8m.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
    args = parser.parse_args()

    bootstrap(args.model)
    app.run(host=args.host, port=args.port, debug=False)
