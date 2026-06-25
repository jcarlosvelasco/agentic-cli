import sys

from dotenv import load_dotenv

from src.shared.config import load_config


def main():
    load_dotenv()
    config = load_config()
    mode = config.mode

    if len(sys.argv) > 1:
        arg_mode = sys.argv[1]
        if arg_mode in ("cli", "api"):
            mode = arg_mode

    if mode == "api":
        import uvicorn

        from src.api.api import app

        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        import asyncio

        from src.shared.main import main as cli_main

        asyncio.run(cli_main())


if __name__ == "__main__":
    main()
