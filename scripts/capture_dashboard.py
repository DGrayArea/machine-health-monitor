"""
Capture a screenshot of the live dashboard, for the README and the report.

Doing this by hand means remembering to log in, wait for the simulator to fill
the chart, size the window the same way each time, and crop consistently. This
does all of that, so screenshots taken weeks apart are actually comparable.

It drives the copy of Google Chrome already installed on the machine, through
Playwright's `channel="chrome"`. No 130 MB Chromium download.

The dashboard needs a login, and the token lives in sessionStorage, which a
plain headless screenshot cannot set. So the script logs in through the API,
writes the token into sessionStorage itself, then reloads.

Usage:
    # with the server already running on :8010
    python scripts/capture_dashboard.py

    # drive the machine into a specific state first
    python scripts/capture_dashboard.py --inject tool_wear --settle 20
    python scripts/capture_dashboard.py --out docs/dashboard-fault.png
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8010"
DEFAULT_OUT = ROOT / "docs" / "dashboard.png"

# Wide enough that the sensor tiles lay out 5-across. Tall enough to reach the
# bottom of the trend chart with a banner and action card showing, which is the
# most informative single frame.
VIEWPORT = {"width": 1440, "height": 1430}


def api(url: str, path: str, token: str | None = None,
        payload: bytes | None = None) -> bytes:
    request = urllib.request.Request(url + path, data=payload)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def wait_for_server(url: str, attempts: int = 20) -> None:
    for _ in range(attempts):
        try:
            api(url, "/api/health")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise SystemExit(
        f"No server answering at {url}.\nStart one with:\n"
        "  uvicorn backend.main:app --port 8010"
    )


def login(url: str) -> str:
    import json
    body = json.dumps({
        "username": config.DEMO_USERNAME,
        "password": config.DEMO_PASSWORD,
    }).encode()
    return json.loads(api(url, "/api/auth/login", payload=body))["access_token"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inject", choices=["overheat", "overload", "tool_wear",
                                             "reset"],
                        help="force a fault before capturing")
    parser.add_argument("--settle", type=float, default=30.0,
                        help="seconds to let the simulator fill the chart")
    # Default to the viewport only. A full-page capture stretches to whatever
    # the alert table happens to be that day, which makes a useless README image
    # and a multi-megabyte file.
    parser.add_argument("--full-page", action="store_true",
                        help="capture the whole scrollable page, not just the viewport")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install playwright")

    wait_for_server(args.url)
    token = login(args.url)

    if args.inject:
        api(args.url, f"/api/simulator/inject/{args.inject}", token=token,
            payload=b"")
        print(f"[sim ] injected {args.inject}")

    if args.settle > 0:
        print(f"[wait] {args.settle:.0f}s for the simulator to build history")
        time.sleep(args.settle)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # channel="chrome" uses the installed Google Chrome rather than
        # downloading Playwright's own Chromium build.
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        # Load once to establish the origin, then plant the token. sessionStorage
        # is per-origin, so this cannot be done before the first navigation.
        page.goto(args.url, wait_until="domcontentloaded")
        page.evaluate("t => sessionStorage.setItem('mhm_token', t)", token)
        page.reload(wait_until="networkidle")

        # Wait for real data rather than a fixed sleep: the banner starts at
        # WAITING and the tiles start at an em dash.
        page.wait_for_function(
            "() => { const b = document.getElementById('banner-status');"
            " return b && b.textContent.trim() !== 'WAITING'; }",
            timeout=20_000,
        )
        page.wait_for_timeout(1500)   # let the chart finish its first paint

        page.screenshot(path=str(args.out), full_page=args.full_page)
        browser.close()

    size_kb = args.out.stat().st_size / 1024
    print(f"[ok  ] wrote {args.out.relative_to(ROOT)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
