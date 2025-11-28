"""Entry point for running the Streamlit dashboard."""

import sys

from streamlit.web import cli as stcli


def main() -> None:
    """Run the Streamlit dashboard."""
    sys.argv = [
        "streamlit",
        "run",
        "hackaton_system/dashboard/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    stcli.main()


if __name__ == "__main__":
    main()
