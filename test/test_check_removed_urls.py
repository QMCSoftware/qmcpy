import sys
import urllib.error
from unittest.mock import patch

from scripts import check_removed_urls as cru

SITE = "https://qmcsoftware.github.io/QMCSoftware/"


def _sitemap(*paths):
    locs = "".join(f"<loc>{SITE}{path}</loc>" for path in paths)
    return f'<?xml version="1.0" encoding="UTF-8"?><urlset>{locs}</urlset>'


def _config(redirect_maps=None):
    plugins = ["material/search", {"mkdocs-jupyter": {"execute": False}}]
    if redirect_maps is not None:
        plugins.append({"redirects": {"redirect_maps": redirect_maps}})
    return {"site_url": SITE, "plugins": plugins}


def _run(tmp_path, monkeypatch, sitemap_paths, redirect_maps=None, extra_argv=()):
    """Run main() offline against a temp sitemap and a temp docs/ tree."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    (docs / "README.md").write_text("home", encoding="utf-8")
    (docs / "good_practices.md").write_text("page", encoding="utf-8")
    sitemap = tmp_path / "sitemap.xml"
    sitemap.write_text(_sitemap(*sitemap_paths), encoding="utf-8")

    monkeypatch.setattr(cru, "read_config", lambda *a, **k: _config(redirect_maps))
    monkeypatch.setattr(sys, "argv", [
        "check_removed_urls.py", "--sitemap", str(sitemap), "--docs-dir", str(docs),
        *extra_argv,
    ])
    return cru.main()


def test_url_path_and_source_round_trip(tmp_path):
    for source, url_path in [("blogs/scipywrapper/index.md", "blogs/scipywrapper/"),
                             ("good_practices.md", "good_practices/"),
                             ("demos/quickstart.ipynb", "demos/quickstart/"),
                             ("index.md", ""), ("README.md", "")]:
        assert cru.url_path_for_source(source) == url_path

    for source in ("README.md", "good_practices.md", "demos/quickstart.ipynb",
                   "api/index.md"):
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("page", encoding="utf-8")
        assert cru.source_exists(cru.url_path_for_source(source), tmp_path)
    assert not cru.source_exists("blogs/scipywrapper/", tmp_path)


def test_redirect_maps_reads_the_plugin_and_tolerates_its_absence():
    entry = {"blogs/x/index.md": "https://qmcsoftware.org/blogs/x/"}
    assert cru.redirect_maps(_config(entry)) == entry
    assert cru.redirect_maps(_config()) == {}
    assert cru.redirect_maps({}) == {}


def test_published_paths_separates_foreign_urls():
    sitemap = _sitemap("", "good_practices/").replace(
        "</urlset>", "<loc>https://example.test/other/</loc></urlset>")

    assert cru.published_paths(sitemap, SITE) == (
        ["", "good_practices/"], ["https://example.test/other/"])


def test_http_status_falls_back_to_get_when_head_is_unsupported():
    url = "https://example.test"
    error = urllib.error.HTTPError(url, 405, "test response", {}, None)
    response = type("Response", (), {"status": 200, "__enter__": lambda s: s,
                                     "__exit__": lambda s, *a: False})()
    with patch.object(cru.urllib.request, "urlopen",
                      side_effect=[error, response]) as urlopen:
        assert cru.http_status(url, timeout=1) == "200"

    assert urlopen.call_count == 2
    assert urlopen.call_args_list[1].args[0].get_method() == "GET"


def test_removed_page_without_redirect_is_flagged(tmp_path, monkeypatch, capsys):
    code = _run(tmp_path, monkeypatch, ["", "good_practices/", "blogs/scipywrapper/"])
    out = capsys.readouterr().out

    assert code == 1
    assert "1 removed with no redirect" in out
    assert f"[ORPHAN]   {SITE}blogs/scipywrapper/" in out
    assert "blogs/scipywrapper/index.md: <new URL or page>" in out


def test_removed_page_covered_by_a_redirect_passes(tmp_path, monkeypatch, capsys):
    code = _run(
        tmp_path, monkeypatch, ["", "good_practices/", "blogs/scipywrapper/"],
        redirect_maps={
            "blogs/scipywrapper/index.md": "https://qmcsoftware.org/blogs/scipywrapper/"},
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "0 removed with no redirect" in out
    assert "[redirect]" in out and "[ORPHAN]" not in out


def test_intact_site_passes(tmp_path, monkeypatch, capsys):
    assert _run(tmp_path, monkeypatch, ["", "good_practices/"]) == 0
    assert "2 still have a page source" in capsys.readouterr().out


def test_verify_redirects_follows_the_target_status(tmp_path, monkeypatch, capsys):
    redirects = {"blogs/x/index.md": "https://qmcsoftware.org/blogs/x/"}
    for status, expected_code in [("200", 0), ("404", 1)]:
        monkeypatch.setattr(cru, "http_status", lambda *a, **k: status)
        code = _run(tmp_path / status, monkeypatch, ["", "blogs/x/"],
                    redirect_maps=redirects, extra_argv=("--verify-redirects",))
        out = capsys.readouterr().out

        assert code == expected_code
        assert status in out
        # The URL itself is covered, so a failure is the target, not an orphan.
        assert "[ORPHAN]" not in out


def test_unreachable_sitemap_fails_unless_offline_is_allowed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cru, "read_config", lambda *a, **k: _config())
    argv = ["check_removed_urls.py", "--sitemap", str(tmp_path / "absent.xml")]

    monkeypatch.setattr(sys, "argv", argv)
    assert cru.main() == 1

    monkeypatch.setattr(sys, "argv", argv + ["--allow-offline"])
    assert cru.main() == 0
    assert "skipping the check" in capsys.readouterr().out
