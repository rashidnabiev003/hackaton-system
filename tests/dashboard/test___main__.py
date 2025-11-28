"""Tests for dashboard entry point."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from hackaton_system.dashboard.__main__ import main


def test_main_sets_sys_argv_correctly() -> None:
    """Test that main() sets sys.argv correctly before calling streamlit."""
    original_argv = sys.argv.copy()

    with patch("hackaton_system.dashboard.__main__.stcli.main") as mock_main:
        main()
        mock_main.assert_called_once()

    # Check that sys.argv was set correctly
    assert sys.argv[0] == "streamlit"
    assert sys.argv[1] == "run"
    assert sys.argv[2] == "hackaton_system/dashboard/app.py"
    assert "--server.port=8501" in sys.argv
    assert "--server.address=0.0.0.0" in sys.argv

    # Restore original argv
    sys.argv = original_argv


def test_main_calls_streamlit_cli() -> None:
    """Test that main() calls streamlit CLI."""
    with patch("hackaton_system.dashboard.__main__.stcli.main") as mock_main:
        main()
        mock_main.assert_called_once()

