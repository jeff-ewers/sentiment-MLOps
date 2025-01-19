import logging
import sys
from typing import Dict, Any
import json

class JsonFormatter(logging.Formatter):
    """Format logs as JSON for better processing"""
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        }

        # add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        return json.dumps(log_record)
    
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup structured logging for the application"""
    logger = logging.getLogger('sentiment-mlops')
    logger.setLevel(level)

    # remove existing handlers
    logger.handlers.clear()

    # console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)

    # file handler with persistent logs
    file_handler = logging.FileHandler(sys.stdout)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger