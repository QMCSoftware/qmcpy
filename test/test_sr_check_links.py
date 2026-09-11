import contextlib
import io
import shutil
import ssl
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts import check_links


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "test response", {}, None)


class TestCheckLinks(unittest.TestCase):

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_path, ignore_errors=True)

    def _patch(self, target, name, value):
        """monkeypatch.setattr equivalent: set now, auto-restore at test end."""
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_head_success_is_reachable(self):
        with patch.object(check_links.urllib.request, "urlopen", return_value=object()) as urlopen:
            self.assertIsNone(check_links._check_one("https://example.test", timeout=1))

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(urlopen.call_args.args[0].get_method(), "HEAD")

    def test_get_success_after_head_failure_is_reachable(self):
        url = "https://example.test"
        with patch.object(
            check_links.urllib.request,
            "urlopen",
            side_effect=[_http_error(url, 405), object()],
        ) as urlopen:
            self.assertIsNone(check_links._check_one(url, timeout=1))

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[1].args[0].get_method(), "GET")

    def test_not_found_and_gone_gets_are_broken(self):
        for code in (404, 410):
            with self.subTest(code=code):
                url = f"https://example.test/{code}"
                with patch.object(
                    check_links.urllib.request,
                    "urlopen",
                    side_effect=[_http_error(url, code), _http_error(url, code)],
                ):
                    self.assertEqual(
                        check_links._check_one(url, timeout=1),
                        ("broken", f"{url} -- HTTP {code}"),
                    )

    def test_bot_block_and_rate_limit_are_warnings(self):
        for code in (403, 429):
            with self.subTest(code=code):
                url = f"https://example.test/{code}"
                with patch.object(
                    check_links.urllib.request,
                    "urlopen",
                    side_effect=[_http_error(url, code), _http_error(url, code)],
                ):
                    severity, message = check_links._check_one(url, timeout=1)

                self.assertEqual(severity, "warning")
                self.assertIn(f"HTTP {code}", message)

    def test_tls_and_timeout_failures_are_warnings(self):
        failures = (
            ssl.SSLCertVerificationError("certificate verify failed"),
            TimeoutError("timed out"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with patch.object(
                    check_links.urllib.request,
                    "urlopen",
                    side_effect=[failure, failure],
                ):
                    severity, message = check_links._check_one(
                        "https://example.test", timeout=1
                    )

                self.assertEqual(severity, "warning")
                self.assertIn(str(failure), message)

    def test_external_results_are_separated_and_duplicate_urls_checked_once(self):
        (self.tmp_path / "page.html").write_text(
            '<a href="https://example.test/missing">missing</a>'
            '<a href="https://example.test/missing">duplicate</a>'
            '<a href="https://example.test/blocked">blocked</a>',
            encoding="utf-8",
        )

        def result_for(url, _timeout):
            if url.endswith("/missing"):
                return "broken", f"{url} -- HTTP 404"
            return "warning", f"{url} -- HTTP 403"

        with patch.object(check_links, "_check_one", side_effect=result_for) as check_one:
            broken, warnings = check_links.check_external(self.tmp_path, workers=1)

        self.assertEqual(check_one.call_count, 2)
        self.assertEqual(
            broken,
            ["https://example.test/missing -- HTTP 404 (seen on page.html)"],
        )
        self.assertEqual(
            warnings,
            ["https://example.test/blocked -- HTTP 403 (seen on page.html)"],
        )

    def test_internal_links_strip_site_url_deployment_path(self):
        target = self.tmp_path / "target"
        target.mkdir()
        (target / "index.html").write_text(
            '<h2 id="section">Target</h2>', encoding="utf-8"
        )
        (self.tmp_path / "index.html").write_text(
            '<a href="/QMCSoftware/target/">root-relative</a>'
            '<a href="https://qmcsoftware.github.io/QMCSoftware/target/#section">absolute</a>',
            encoding="utf-8",
        )

        self.assertEqual(
            check_links.check_internal(
                self.tmp_path, site_url="https://qmcsoftware.github.io/QMCSoftware/"
            ),
            [],
        )

    def test_external_check_skips_same_site_urls(self):
        (self.tmp_path / "page.html").write_text(
            '<a href="https://qmcsoftware.github.io/QMCSoftware/target/">same</a>'
            '<a href="https://example.test/target/">external</a>',
            encoding="utf-8",
        )

        with patch.object(check_links, "_check_one", return_value=None) as check_one:
            broken, warnings = check_links.check_external(
                self.tmp_path,
                workers=1,
                site_url="https://qmcsoftware.github.io/QMCSoftware/",
            )

        self.assertEqual(broken, [])
        self.assertEqual(warnings, [])
        self.assertEqual(check_one.call_count, 1)
        self.assertEqual(check_one.call_args.args[0], "https://example.test/target/")

    def test_external_warnings_do_not_make_main_fail(self):
        self._patch(sys, "argv", ["check_links.py", str(self.tmp_path), "--external"])
        self._patch(
            check_links, "check_internal", lambda _site_dir, site_url=None: []
        )
        self._patch(
            check_links,
            "check_external",
            lambda _site_dir, site_url=None: (
                [],
                ["https://example.test -- HTTP 403"],
            ),
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(check_links.main(), 0)
        self.assertIn("0 broken link(s), 1 warning(s)", buf.getvalue())

    def test_confirmed_external_breakage_makes_main_fail(self):
        self._patch(sys, "argv", ["check_links.py", str(self.tmp_path), "--external"])
        self._patch(
            check_links, "check_internal", lambda _site_dir, site_url=None: []
        )
        self._patch(
            check_links,
            "check_external",
            lambda _site_dir, site_url=None: (
                ["https://example.test -- HTTP 404"],
                [],
            ),
        )

        self.assertEqual(check_links.main(), 1)


if __name__ == "__main__":
    unittest.main()
