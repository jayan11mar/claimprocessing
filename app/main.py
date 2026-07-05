import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from app.config import get_settings
from app.api.server import app as fastapi_app

app = fastapi_app


def main() -> None:
    settings = get_settings()
    port = int(os.getenv("API_PORT", "8000"))
    print("Claims Processing & Settlement.")
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)


if __name__ == "__main__":
    main()
