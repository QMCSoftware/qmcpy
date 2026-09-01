import ssl
import sys
import urllib.error
from unittest.mock import patch

from scripts import check_links


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "test response", {}, None)


def test_head_success_is_reachable():
    with patch.object(check_links.urllib.request, "urlopen", return_value=object()) as urlopen:
        assert check_links._check_one("https://example.test", timeout=1) is None

    assert urlopen.call_count == 1
    assert urlopen.call_args.args[0].get_method() == "HEAD"


def test_get_success_after_head_failure_is_reachable():
    url = "https://example.test"
    with patch.object(
        check_links.urllib.request,
        "urlopen",
        side_effect=[_http_error(url, 405), object()],
    ) as urlopen:
        assert check_links._check_one(url, timeout=1) is None

    assert urlopen.call_count == 2
    assert urlopen.call_args_list[1].args[0].get_method() == "GET"


def test_not_found_and_gone_gets_are_broken():
    for code in (404, 410):
        url = f"https://example.test/{code}"
        with patch.object(
            check_links.urllib.request,
            "urlopen",
            side_effect=[_http_error(url, code), _http_error(url, code)],
        ):
            assert check_links._check_one(url, timeout=1) == (
                "broken",
                f"{url} -- HTTP {code}",
            )


def test_bot_block_and_rate_limit_are_warnings():
    for code in (403, 429):
        url = f"https://example.test/{code}"
        with patch.object(
            check_links.urllib.request,
            "urlopen",
            side_effect=[_http_error(url, code), _http_error(url, code)],
        ):
            severity, message = check_links._check_one(url, timeout=1)

        assert severity == "warning"
        assert f"HTTP {code}" in message


def test_tls_and_timeout_failures_are_warnings():
    failures = (
        ssl.SSLCertVerificationError("certificate verify failed"),
        TimeoutError("timed out"),
    )
    for failure in failures:
        with patch.object(
            check_links.urllib.request,
            "urlopen",
            side_effect=[failure, failure],
        ):
            severity, message = check_links._check_one(
                "https://example.test", timeout=1
            )

        assert severity == "warning"
        assert str(failure) in message


def test_external_results_are_separated_and_duplicate_urls_checked_once(tmp_path):
    (tmp_path / "page.html").write_text(
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
        broken, warnings = check_links.check_external(tmp_path, workers=1)

    assert check_one.call_count == 2
    assert broken == [
        "https://example.test/missing -- HTTP 404 (seen on page.html)"
    ]
    assert warnings == [
        "https://example.test/blocked -- HTTP 403 (seen on page.html)"
    ]


def test_internal_links_strip_site_url_deployment_path(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "index.html").write_text(
        '<h2 id="section">Target</h2>', encoding="utf-8"
    )
    (tmp_path / "index.html").write_text(
        '<a href="/QMCSoftware/target/">root-relative</a>'
        '<a href="https://qmcsoftware.github.io/QMCSoftware/target/#section">absolute</a>',
        encoding="utf-8",
    )

    assert (
        check_links.check_internal(
            tmp_path, site_url="https://qmcsoftware.github.io/QMCSoftware/"
        )
        == []
    )


def test_external_check_skips_same_site_urls(tmp_path):
    (tmp_path / "page.html").write_text(
        '<a href="https://qmcsoftware.github.io/QMCSoftware/target/">same</a>'
        '<a href="https://example.test/target/">external</a>',
        encoding="utf-8",
    )

    with patch.object(check_links, "_check_one", return_value=None) as check_one:
        broken, warnings = check_links.check_external(
            tmp_path,
            workers=1,
            site_url="https://qmcsoftware.github.io/QMCSoftware/",
        )

    assert broken == []
    assert warnings == []
    assert check_one.call_count == 1
    assert check_one.call_args.args[0] == "https://example.test/target/"


def test_external_warnings_do_not_make_main_fail(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_links.py", str(tmp_path), "--external"])
    monkeypatch.setattr(
        check_links, "check_internal", lambda _site_dir, site_url=None: []
    )
    monkeypatch.setattr(
        check_links,
        "check_external",
        lambda _site_dir, site_url=None: (
            [],
            ["https://example.test -- HTTP 403"],
        ),
    )

    assert check_links.main() == 0
    assert "0 broken link(s), 1 warning(s)" in capsys.readouterr().out


def test_confirmed_external_breakage_makes_main_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_links.py", str(tmp_path), "--external"])
    monkeypatch.setattr(
        check_links, "check_internal", lambda _site_dir, site_url=None: []
    )
    monkeypatch.setattr(
        check_links,
        "check_external",
        lambda _site_dir, site_url=None: (
            ["https://example.test -- HTTP 404"],
            [],
        ),
    )

    assert check_links.main() == 1
