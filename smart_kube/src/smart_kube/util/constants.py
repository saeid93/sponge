LOGGING_LEVEL: str = 'info'
LOG_TO = 'print'

LIMIT_RANGE = {
    "max": {
        "memory": 800000,
        "cpu": 100000
        },
    "min": {
        "memory": 0,
        "cpu": 0
        },
    "max_limit_request_ratio": {
        "memory": 10,
        "cpu": 10
        }
}