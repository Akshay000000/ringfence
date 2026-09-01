"""Render the social preview card from its HTML source.

The card is a PNG in the end, but a PNG is not editable by anyone who comes
after, so the source of truth is `site/assets/og-template.html` and this script
bakes it. Run it after changing the template:

    python tools/render_og.py

Exactly 1200x630 at 2x, which is what every unfurler crops to.
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "site" / "assets" / "og-template.html"
OUT = ROOT / "site" / "assets" / "og.png"

WIDTH, HEIGHT = 1200, 630


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,
        )
        page.goto(TEMPLATE.as_uri())
        page.wait_for_timeout(400)
        page.screenshot(path=OUT, clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT})
        browser.close()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
