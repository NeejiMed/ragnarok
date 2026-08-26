# Monitoring Stack

## Services
- **Prometheus**: scrapes metrics from the backend `/metrics` endpoint every 15s
- **Grafana**: visualizes Prometheus metrics via dashboards

## Setup
1. Start the full stack: `make up`
2. Open Grafana at `http://localhost:3000` (admin / ragnarok)
3. Add Prometheus data source: Connections → Data sources → Prometheus → URL: `http://prometheus:9090`
4. Import `grafana-dashboard.json` via Dashboards → Import, or build manually

## Key metrics to dashboard
- `ragnarok_requests_total` — requests per minute by endpoint
- `ragnarok_request_duration_seconds` — p50/p95/p99 latency
- `ragnarok_documents_ingested_total` — documents uploaded over time
- `ragnarok_requests_in_progress` — currently active requests