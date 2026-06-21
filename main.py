from app.config import get_settings
from app.api.server import app as fastapi_app

app = fastapi_app


def main() -> None:
    settings = get_settings()
    print("Claims Processing Assistant.")

