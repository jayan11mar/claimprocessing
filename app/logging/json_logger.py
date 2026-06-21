import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "timestamp_ms": int(record.created * 1000),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        try:
            if record.args and isinstance(record.args, dict):
                for k, v in record.args.items():
                    if k not in payload:
                        payload[k] = v
            elif record.args:
                payload["args"] = record.args
        except Exception:
            payload["args"] = str(record.args)

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process"}
        }
        for k in list(extra.keys()):
            if k in payload:
                extra.pop(k, None)

        if extra:
            payload.update(extra)

        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
