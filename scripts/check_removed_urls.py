#!/usr/bin/env python3
"""Check that every already-published URL still resolves after a docs change.

`check_links.py` validates the links *inside* the site we are about to publish.
It cannot see the opposite failure: a deleted page 404s for the search engines,
papers, and other sites still linking to it, and the commit that deleted the
page deleted its links too -- so no crawl of the new site has anything to flag.

This closes that gap by diffing URL inventories instead of links: fetch the
deployed `sitemap.xml` (every URL currently published), map each URL back to
its page source under `docs/`, and report those whose source is gone and which
no `redirect_maps` entry of the mkdocs-redirects plugin covers. Each one is live
today and 404s after the next deploy; fix it by restoring the source, or by
adding a `redirects` entry in mkdocs.yml when the content moved.

Usage:
    make copydocs                                   # docs/ must be populated
    python scripts/check_removed_urls.py
    python scripts/check_removed_urls.py --verify-redirects   # HTTP-check targets
    python scripts/check_removed_urls.py --sitemap site/sitemap.xml   # offline
"""
from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
MKDOCS_CONFIG = ROOT / "mkdocs.yml"
DOCS_DIR = ROOT / "docs"
LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
# Same browser UA as check_links.py: some hosts reject scripted-looking requests.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class _ConfigLoader(yaml.SafeLoader):
    """Tolerates the custom tags mkdocs.yml allows (`!ENV`, `!!python/name:`).

    Subclassed so registering these constructors cannot change how
    `yaml.safe_load` behaves for anything else in the process.
    """


_ConfigLoader.add_multi_constructor("!", lambda *_: None)
_ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda *_: None)


def read_config(config_path: Path = MKDOCS_CONFIG) -> dict:
    """Load mkdocs.yml as a plain dict."""
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
    return config if isinstance(config, dict) else {}


def site_url(config: dict) -> str:
    """`site_url` with a trailing slash; the deployed site's base."""
    url = str(config.get("site_url") or "").strip()
    if not url:
        raise SystemExit("mkdocs.yml has no site_url; cannot locate the deployed sitemap.")
    return url.rstrip("/") + "/"


def redirect_maps(config: dict) -> dict[str, str]:
    """`redirect_maps` from the mkdocs-redirects plugin ({} when not configured)."""
    for plugin in config.get("plugins") or []:
        if isinstance(plugin, dict) and "redirects" in plugin:
            return dict((plugin["redirects"] or {}).get("redirect_maps") or {})
    return {}


def url_path_for_source(source: str) -> str:
    """Page source -> the URL path MkDocs publishes it at, per `use_directory_urls`.

    `a/b.md` -> `a/b/`, `a/index.md` -> `a/`, and the site index -> `` (root).
    """
    path = re.sub(r"\.(md|ipynb)$", "", source.strip().lstrip("/"))
    if path in ("index", "README"):
        return ""
    path = re.sub(r"/index$", "", path)
    return path + "/" if path else ""


def source_exists(url_path: str, docs_dir: Path = DOCS_DIR) -> bool:
    """Whether `url_path` still has a page source: the inverse of the mapping above.

    A URL can come from `<path>.md`, `<path>.ipynb`, or either as `<path>/index.*`;
    the site root additionally comes from `README.md` (this project's homepage).
    """
    stem = url_path.strip("/")
    names = ("index.md", "README.md") if not stem else (
        f"{stem}/index.md", f"{stem}/index.ipynb", f"{stem}.md", f"{stem}.ipynb")
    return any((docs_dir / name).exists() for name in names)


def published_paths(sitemap_xml: str, base: str) -> tuple[list[str], list[str]]:
    """Split sitemap entries into paths under `base` and unexpected foreign URLs."""
    local, foreign = [], []
    for loc in LOC_RE.findall(sitemap_xml):
        if loc.startswith(base):
            local.append(loc[len(base):])
        elif urlparse(loc).scheme in ("http", "https"):
            foreign.append(loc)
    return local, foreign


def http_status(url: str, timeout: float = 15.0) -> str:
    """HTTP status code for `url` as a string ('ERR' when unreachable).

    HEAD first, then GET, since some hosts do not implement HEAD.
    """
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(
            url, headers={"User-Agent": BROWSER_UA}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return str(response.status)
        except urllib.error.HTTPError as error:
            if method == "GET" or error.code not in (403, 404, 405, 501):
                return str(error.code)
        except Exception as error:  # network/TLS/DNS failure
            if method == "GET":
                return f"ERR ({type(error).__name__})"
    return "ERR"


def read_sitemap(source: str, timeout: float) -> str:
    """Read a sitemap from a local path or over HTTP."""
    if urlparse(source).scheme not in ("http", "https"):
        return Path(source).read_text(encoding="utf-8")
    request = urllib.request.Request(source, headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sitemap", help="sitemap.xml path or URL to use instead of "
                                          "the deployed one")
    parser.add_argument("--docs-dir", default=str(DOCS_DIR),
                        help="page-source directory to check against (default: docs)")
    parser.add_argument("--config", default=str(MKDOCS_CONFIG),
                        help="MkDocs config to read site_url and redirect_maps from")
    parser.add_argument("--verify-redirects", action="store_true",
                        help="also HTTP-check that every redirect target resolves")
    parser.add_argument("--timeout", type=float, default=30.0, help="network timeout (s)")
    parser.add_argument("--allow-offline", action="store_true",
                        help="exit 0 instead of 1 when the sitemap cannot be read")
    args = parser.parse_args()

    config = read_config(Path(args.config))
    base = site_url(config)
    docs_dir = Path(args.docs_dir)
    redirects = redirect_maps(config)
    covered = {url_path_for_source(src): target for src, target in redirects.items()}

    source = args.sitemap or f"{base}sitemap.xml"
    try:
        sitemap_xml = read_sitemap(source, args.timeout)
    except Exception as error:
        print(f"Could not read {source}: {error}")
        if args.allow_offline:
            print("  --allow-offline given, skipping the check.")
            return 0
        return 1

    paths, foreign = published_paths(sitemap_xml, base)
    if not paths:
        print(f"No URLs under {base} found in {source} -- is site_url correct?")
        return 1

    missing = [p for p in paths if not source_exists(p, docs_dir)]
    orphans = sorted(p for p in missing if p not in covered)
    handled = sorted(p for p in missing if p in covered)

    print(f"Checked {len(paths)} published URL(s) from {source} against {docs_dir}/:\n"
          f"  {len(paths) - len(missing)} still have a page source\n"
          f"  {len(handled)} removed and covered by a redirect\n"
          f"  {len(orphans)} removed with no redirect")
    for path in handled:
        print(f"  [redirect] {base}{path} -> {covered[path]}")
    for path in orphans:
        print(f"  [ORPHAN]   {base}{path} -- live now, 404 after the next deploy")
    for loc in sorted(foreign):
        print(f"  [warning] sitemap lists a URL outside {base}: {loc}")
    if orphans:
        print("\nRestore each ORPHAN's page source, or -- if the content moved -- add it "
              "to\nthe `redirects` plugin's redirect_maps in mkdocs.yml:")
        for path in orphans:
            print(f"        {path}index.md: <new URL or page>")

    problems = list(orphans)
    if args.verify_redirects and redirects:
        print(f"\nVerifying {len(redirects)} redirect target(s):")
        for target in sorted(set(redirects.values())):
            external = urlparse(target).scheme in ("http", "https")
            status = (http_status(target, args.timeout) if external else
                      "ok" if source_exists(url_path_for_source(target), docs_dir)
                      else "MISS")
            print(f"  {status:>5}  {target}")
            if not (status == "ok" or status.startswith(("2", "3"))):
                problems.append(f"redirect target does not resolve: {target} ({status})")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
