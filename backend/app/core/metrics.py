from prometheus_client import Counter, Gauge, Histogram

# Total requests by method, endpoint and HTTP status code
REQUEST_COUNT = Counter(
    "ragnarok_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"]
)

# Request duration in seconds, histogram gives us p50/p90/p99
REQUEST_LATENCY = Histogram(
    "ragnarok_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# currently active requests, gauge because it can go up and down
REQUEST_IN_PROGRESS = Gauge(
    "ragnarok_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "endpoint"]
)

# Business metric, total documents successfully ingested
DOCUMENTS_INGESTED = Counter(
    "ragnarok_documents_ingested_total",
    "Total number of documents successfully ingested",
    ["document_type"]
)