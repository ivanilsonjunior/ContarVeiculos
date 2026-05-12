# Natal Traffic Monitor 🚗

Monitor de tráfego em tempo real usando YOLO na webcam do Monza Palace Hotel — Natal/RN.

## Estrutura

```
natal_traffic/
├── app.py            ← Servidor Flask + endpoints da API
├── detector.py       ← Thread de captura e detecção YOLO
├── requirements.txt
└── templates/
    └── index.html    ← Dashboard ao vivo
```

## Instalação

```bash
pip install -r requirements.txt
```

> O modelo YOLOv8 (`yolov8n.pt`) é baixado automaticamente na primeira execução.

## Executar

```bash
# Padrão (localhost:5000, modelo nano)
python app.py

# Personalizado
python app.py --host 0.0.0.0 --port 8080 --model yolov8s.pt
```

Acesse: **http://localhost:5000**

---

## Endpoints da API

### `GET /api/stats`
Retorna as métricas principais.

```json
{
  "status": "ok",
  "timestamp": "2026-05-11T22:00:00+00:00",
  "avg_last_minute":  3.4,
  "moving_avg_10min": 2.8,
  "last_sample": {
    "ts": "2026-05-11T21:59:50+00:00",
    "total": 4,
    "car": 3,
    "motorcycle": 1,
    "bus": 0,
    "truck": 0
  },
  "history_10min": [
    { "ts": "2026-05-11T21:50:00+00:00", "avg": 2.6 },
    { "ts": "2026-05-11T21:51:00+00:00", "avg": 3.1 }
  ]
}
```

### `GET /api/history`
Retorna todas as amostras brutas (últimas 2 horas).

Query params:
- `?limit=N` — limita ao N mais recentes

```json
{
  "count": 42,
  "samples": [
    { "ts": "...", "total": 4, "car": 3, "motorcycle": 1, "bus": 0, "truck": 0 }
  ]
}
```

### `GET /api/snapshot`
Retorna o último frame anotado como `image/jpeg`.

### `GET /api/status`
Saúde do serviço.

```json
{
  "status": "ok",
  "samples_count": 42
}
```

---

## Como funciona

```
PLAYER_URL (HTML)
      │
      ▼
detector.py (thread daemon)
  ├── Tenta extrair URL HLS/m3u8 do player
  ├── Fallback: parse do <img src> do HTML do player
  │
  ▼ frame capturado a cada 10s
  │
YOLOv8
  └── Detecta: car / motorcycle / bus / truck
         │
         ▼
    _raw (deque, últimas 2h)
    _min_avgs (deque, últimos 10 minutos)
         │
         ▼
Flask /api/stats → Dashboard + API consumers
```

## Notas

- A câmera atualiza o feed a cada poucos minutos; capturas mais frequentes podem retornar o mesmo frame.
- A média móvel de 10 minutos é calculada sobre as médias-por-minuto (não sobre amostras brutas).
- O modelo `yolov8n` é o mais rápido; use `yolov8m` ou `yolov8l` para maior precisão.
