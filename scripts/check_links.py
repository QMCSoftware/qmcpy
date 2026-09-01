#!/usr/bin/env python3
"""Check for broken links in a built MkDocs site.

Crawls the static HTML output of `mkdocs build` and reports:
- internal links/anchors that don't resolve to a page or heading in the site
  (always checked; these are fully within our control and any failure here
  is a real build defect)
- external links that return an error or fail to connect (only checked with
  --external, since third-party sites rate-limit, gate behind auth walls, or
  have transient outages that should not fail a build)

Usage:
    mkdocs build -d site
    python scripts/check_links.py site
    python scripts/check_links.py site --external
"""
from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml

HREF_RE = re.compile(r'href=["\']([^"\'#][^"\']*)["\']')
# Only <a href="...">, not <link href="..."> (stylesheets, preconnect resource
# hints, icons, ...) -- those aren't links a reader can click and follow.
A_HREF_RE = re.compile(r'<a\s[^>]*href=["\']([^"\'#][^"\']*)["\']', re.IGNORECASE)
ID_RE = re.compile(r'id=["\']([^"\']+)["\']')
ROOT = Path(__file__).resolve().parents[1]
MKDOCS_CONFIG = ROOT / "mkdocs.yml"


def read_site_url(config_path: Path = MKDOCS_CONFIG) -> str | None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    value = config.get("site_url") if isinstance(config, dict) else None
    if value is None:
        return None
    return str(value).strip() or None


def _local_href(href: str, site_url: str | None) -> str | None:
    """Return a site-local href, stripping the deployment path when needed."""
    parsed = urlparse(href)
    site = urlparse(site_url) if site_url else None

    if parsed.scheme:
        if (
            site is None
            or parsed.scheme not in ("http", "https")
            or (parsed.scheme, parsed.netloc) != (site.scheme, site.netloc)
        ):
            return None
    elif href.startswith("//"):
        return None

    path = parsed.path
    prefix = site.path.rstrip("/") if site is not None else ""
    if prefix and (path == prefix or path.startswith(f"{prefix}/")):
        path = path[len(prefix) :] or "/"

    return urlunparse(("", "", path, "", parsed.query, parsed.fragment))


def _path_for_url_target(site_dir: Path, target: str) -> Path | None:
    """Map a site-relative href to the file it should resolve to, mirroring
    MkDocs's `use_directory_urls` output layout (page.md -> page/index.html)."""
    clean = target.split("#", 1)[0].split("?", 1)[0]
    if not clean or clean == ".":
        return None
    # site_dir is already resolved by the caller, so .resolve() here only
    # normalizes "../" segments -- it does not re-translate any symlink.
    candidate = (site_dir / clean.lstrip("/")).resolve()
    if candidate.is_dir():
        candidate = candidate / "index.html"
    elif candidate.suffix == "":
        alt = candidate.parent / (candidate.name + "/index.html")
        if alt.exists():
            candidate = alt
    return candidate


def check_internal(site_dir: Path, site_url: str | None = None) -> list[str]:
    # Resolve once so every path derived from site_dir (including each html
    # file's .parent used as the base for relative links) shares one
    # convention -- otherwise a symlinked path (e.g. macOS /tmp -> /private/tmp)
    # makes filesystem-equal paths compare unequal as dict keys.
    site_dir = site_dir.resolve()
    problems = []
    html_files = sorted(site_dir.rglob("*.html"))
    if not html_files:
        return [f"no .html files found under {site_dir} -- did `mkdocs build -d {site_dir}` run first?"]

    anchors_by_file: dict[Path, set[str]] = {}
    for f in html_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        anchors_by_file[f] = set(ID_RE.findall(text))

    for f in html_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for href in HREF_RE.findall(text):
            local_href = _local_href(href, site_url)
            if local_href is None:
                continue
            base_dir = site_dir if local_href.startswith("/") else f.parent
            target_path = _path_for_url_target(base_dir, local_href)
            if target_path is None:
                continue
            try:
                exists = target_path.exists()
            except OSError as e:
                problems.append(f"{f.relative_to(site_dir)}: unresolvable link '{href[:80]}...' ({e})")
                continue
            if not exists:
                problems.append(
                    f"{f.relative_to(site_dir)}: broken internal link '{href}'"
                )
                continue
            if "#" in local_href:
                anchor = local_href.split("#", 1)[1]
                if anchor and anchor not in anchors_by_file.get(target_path, set()):
                    problems.append(
                        f"{f.relative_to(site_dir)}: link '{href}' has no matching "
                        f"id='{anchor}' on the target page"
                    )
    return problems


# Some hosts (Figma confirmed) 404 any HEAD request regardless of who's
# asking, but 200 a GET from a browser-like UA -- their HEAD route is simply
# unimplemented, not a bot check. Others (ACM, Wiley, INFORMS, ScienceDirect,
# MathWorks, SigOpt, doi.org, ...) 403 real browsers too on scripted-looking
# requests; a GET retry there confirms rather than changes the outcome. So:
# always retry any HEAD failure with a browser-UA GET, and only label it
# "likely bot-blocked" if that GET *also* comes back 403.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _check_one(url: str, timeout: float) -> tuple[str, str] | None:
    """Return ``(severity, message)`` when a URL cannot be verified.

    A 404 or 410 returned by a browser-like GET is a confirmed broken link.
    Access controls, rate limits, server errors, redirects that urllib cannot
    follow, and network/TLS failures are warnings: they do not prove that the
    target is broken and should not make a documentation build fail.
    """
    try:
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": BROWSER_UA}, method="HEAD"),
            timeout=timeout,
        )
        return None
    except Exception:
        pass

    try:
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": BROWSER_UA}),
            timeout=timeout,
        )
        return None
    except urllib.error.HTTPError as e:
        if e.code in (404, 410):
            return "broken", f"{url} -- HTTP {e.code}"
        if e.code == 403:
            return "warning", f"[likely bot-blocked, verify manually] {url} -- HTTP 403"
        return "warning", f"{url} -- HTTP {e.code}"
    except Exception as e:
        return "warning", f"{url} -- {e}"


def check_external(
    site_dir: Path,
    timeout: float = 8.0,
    workers: int = 4,
    site_url: str | None = None,
) -> tuple[list[str], list[str]]:
    links: dict[str, list[Path]] = {}
    for f in sorted(site_dir.rglob("*.html")):
        text = f.read_text(encoding="utf-8", errors="replace")
        for href in A_HREF_RE.findall(text):
            if (
                urlparse(href).scheme in ("http", "https")
                and _local_href(href, site_url) is None
            ):
                links.setdefault(href, []).append(f.relative_to(site_dir))

    broken = []
    warnings = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_url = {pool.submit(_check_one, url, timeout): url for url in links}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            result = future.result()
            if result is not None:
                severity, message = result
                report = f"{message} (seen on {links[url][0]})"
                if severity == "broken":
                    broken.append(report)
                else:
                    warnings.append(report)
    return sorted(broken), sorted(warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", nargs="?", default="site", help="MkDocs build output directory")
    parser.add_argument("--external", action="store_true", help="also check external (http/https) links")
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    site_url = read_site_url()
    problems = check_internal(site_dir, site_url=site_url)
    print(f"Checked internal links under {site_dir}: {len(problems)} problem(s).")
    for p in problems:
        print(f"  {p}")

    if args.external:
        ext_broken, ext_warnings = check_external(site_dir, site_url=site_url)
        print(
            f"\nChecked external links: {len(ext_broken)} broken link(s), "
            f"{len(ext_warnings)} warning(s)."
        )
        for p in ext_broken:
            print(f"  {p}")
        for p in ext_warnings:
            print(f"  [warning] {p}")
        problems += ext_broken

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
