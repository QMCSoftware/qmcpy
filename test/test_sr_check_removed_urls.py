import contextlib
import io
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
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


class TestCheckRemovedUrls(unittest.TestCase):

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_path, ignore_errors=True)
        self._last_out = ""

    def _patch(self, target, name, value):
        """monkeypatch.setattr equivalent: set now, auto-restore at test end."""
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, sitemap_paths, redirect_maps=None, extra_argv=(), base=None):
        """Run main() offline against a temp sitemap and a temp docs/ tree."""
        base = self.tmp_path if base is None else base
        docs = base / "docs"
        docs.mkdir(parents=True)
        (docs / "README.md").write_text("home", encoding="utf-8")
        (docs / "good_practices.md").write_text("page", encoding="utf-8")
        sitemap = base / "sitemap.xml"
        sitemap.write_text(_sitemap(*sitemap_paths), encoding="utf-8")

        self._patch(cru, "read_config", lambda *a, **k: _config(redirect_maps))
        self._patch(sys, "argv", [
            "check_removed_urls.py", "--sitemap", str(sitemap), "--docs-dir", str(docs),
            *extra_argv,
        ])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cru.main()
        self._last_out = buf.getvalue()
        return code

    def test_url_path_and_source_round_trip(self):
        for source, url_path in [("blogs/scipywrapper/index.md", "blogs/scipywrapper/"),
                                 ("good_practices.md", "good_practices/"),
                                 ("demos/quickstart.ipynb", "demos/quickstart/"),
                                 ("index.md", ""), ("README.md", "")]:
            self.assertEqual(cru.url_path_for_source(source), url_path)

        for source in ("README.md", "good_practices.md", "demos/quickstart.ipynb",
                       "api/index.md"):
            path = self.tmp_path / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("page", encoding="utf-8")
            self.assertTrue(
                cru.source_exists(cru.url_path_for_source(source), self.tmp_path)
            )
        self.assertFalse(cru.source_exists("blogs/scipywrapper/", self.tmp_path))

    def test_redirect_maps_reads_the_plugin_and_tolerates_its_absence(self):
        entry = {"blogs/x/index.md": "https://qmcsoftware.org/blogs/x/"}
        self.assertEqual(cru.redirect_maps(_config(entry)), entry)
        self.assertEqual(cru.redirect_maps(_config()), {})
        self.assertEqual(cru.redirect_maps({}), {})

    def test_published_paths_separates_foreign_urls(self):
        sitemap = _sitemap("", "good_practices/").replace(
            "</urlset>", "<loc>https://example.test/other/</loc></urlset>")

        self.assertEqual(
            cru.published_paths(sitemap, SITE),
            (["", "good_practices/"], ["https://example.test/other/"]),
        )

    def test_http_status_falls_back_to_get_when_head_is_unsupported(self):
        url = "https://example.test"
        error = urllib.error.HTTPError(url, 405, "test response", {}, None)
        response = type("Response", (), {"status": 200, "__enter__": lambda s: s,
                                         "__exit__": lambda s, *a: False})()
        with patch.object(cru.urllib.request, "urlopen",
                          side_effect=[error, response]) as urlopen:
            self.assertEqual(cru.http_status(url, timeout=1), "200")

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[1].args[0].get_method(), "GET")

    def test_removed_page_without_redirect_is_flagged(self):
        code = self._run(["", "good_practices/", "blogs/scipywrapper/"])
        out = self._last_out

        self.assertEqual(code, 1)
        self.assertIn("1 removed with no redirect", out)
        self.assertIn(f"[ORPHAN]   {SITE}blogs/scipywrapper/", out)
        self.assertIn("blogs/scipywrapper/index.md: <new URL or page>", out)

    def test_removed_page_covered_by_a_redirect_passes(self):
        code = self._run(
            ["", "good_practices/", "blogs/scipywrapper/"],
            redirect_maps={
                "blogs/scipywrapper/index.md": "https://qmcsoftware.org/blogs/scipywrapper/"},
        )
        out = self._last_out

        self.assertEqual(code, 0)
        self.assertIn("0 removed with no redirect", out)
        self.assertIn("[redirect]", out)
        self.assertNotIn("[ORPHAN]", out)

    def test_intact_site_passes(self):
        self.assertEqual(self._run(["", "good_practices/"]), 0)
        self.assertIn("2 still have a page source", self._last_out)

    def test_verify_redirects_follows_the_target_status(self):
        redirects = {"blogs/x/index.md": "https://qmcsoftware.org/blogs/x/"}
        for status, expected_code in [("200", 0), ("404", 1)]:
            with self.subTest(status=status):
                self._patch(cru, "http_status", lambda *a, **k: status)
                code = self._run(
                    ["", "blogs/x/"],
                    redirect_maps=redirects,
                    extra_argv=("--verify-redirects",),
                    base=self.tmp_path / status,
                )
                out = self._last_out

                self.assertEqual(code, expected_code)
                self.assertIn(status, out)
                # The URL itself is covered, so a failure is the target, not an orphan.
                self.assertNotIn("[ORPHAN]", out)

    def test_unreachable_sitemap_fails_unless_offline_is_allowed(self):
        self._patch(cru, "read_config", lambda *a, **k: _config())
        argv = ["check_removed_urls.py", "--sitemap", str(self.tmp_path / "absent.xml")]

        self._patch(sys, "argv", argv)
        self.assertEqual(cru.main(), 1)

        self._patch(sys, "argv", argv + ["--allow-offline"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cru.main(), 0)
        self.assertIn("skipping the check", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
