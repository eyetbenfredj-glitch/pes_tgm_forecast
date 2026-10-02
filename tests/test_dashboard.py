import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

PAGES = [
    "1. National overview", "2. Governorate overview", "3. District forecast",
    "4. Uncertainty", "5. Forecast vs observed", "6. Forecast errors",
    "7. Weather", "8. Monitoring / anomalies", "9. Model / explainability",
    "10. Data provenance / status", "Tunisia map",
]


@pytest.mark.parametrize("page", PAGES)
def test_dashboard_page_renders_without_exception(page):
    app_path = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    at.radio[0].set_value(page).run()
    assert len(at.exception) == 0, f"{page} raised: {[str(e) for e in at.exception]}"
