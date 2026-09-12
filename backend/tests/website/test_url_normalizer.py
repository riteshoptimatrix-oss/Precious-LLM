"""
Unit Tests — URLNormalizer

Covers: domain restriction, SSRF defense, fragment stripping,
tracking param removal, relative URL resolution, scheme enforcement.
"""

import pytest
from app.website.crawler.url_normalizer import URLNormalizer


@pytest.fixture
def normalizer():
    return URLNormalizer()


class TestIsAllowedDomain:
    def test_www_subdomain_allowed(self, normalizer):
        assert normalizer.is_allowed_domain("https://www.preciousedu.in/services") is True

    def test_bare_domain_allowed(self, normalizer):
        assert normalizer.is_allowed_domain("https://preciousedu.in/about") is True

    def test_external_domain_blocked(self, normalizer):
        assert normalizer.is_allowed_domain("https://google.com/search") is False

    def test_ssrf_localhost_blocked(self, normalizer):
        assert normalizer.is_allowed_domain("http://localhost:8080/admin") is False

    def test_ssrf_internal_ip_blocked(self, normalizer):
        assert normalizer.is_allowed_domain("http://192.168.1.1/") is False

    def test_subdomain_attack_blocked(self, normalizer):
        assert normalizer.is_allowed_domain("https://evil.preciousedu.in/") is False

    def test_empty_string_returns_false(self, normalizer):
        assert normalizer.is_allowed_domain("") is False


class TestNormalize:
    def test_strips_fragment(self, normalizer):
        result = normalizer.normalize("https://www.preciousedu.in/about#team")
        assert result == "https://www.preciousedu.in/about"

    def test_strips_tracking_params(self, normalizer):
        result = normalizer.normalize(
            "https://www.preciousedu.in/services?utm_source=google&utm_medium=cpc"
        )
        assert "utm_source" not in (result or "")
        assert "utm_medium" not in (result or "")

    def test_preserves_legit_query_params(self, normalizer):
        result = normalizer.normalize("https://www.preciousedu.in/search?q=visa")
        assert result is not None
        assert "q=visa" in result

    def test_relative_url_resolves(self, normalizer):
        result = normalizer.normalize("/contact", base_url="https://www.preciousedu.in/")
        assert result == "https://www.preciousedu.in/contact"

    def test_external_url_returns_none(self, normalizer):
        result = normalizer.normalize("https://openai.com/chat")
        assert result is None

    def test_mailto_returns_none(self, normalizer):
        result = normalizer.normalize("mailto:info@preciousedu.in")
        assert result is None

    def test_javascript_returns_none(self, normalizer):
        result = normalizer.normalize("javascript:void(0)")
        assert result is None

    def test_trailing_slash_normalized(self, normalizer):
        result = normalizer.normalize("https://www.preciousedu.in/services/")
        assert result == "https://www.preciousedu.in/services"

    def test_root_path_preserved(self, normalizer):
        result = normalizer.normalize("https://www.preciousedu.in/")
        assert result is not None
        assert result.endswith("/") or result == "https://www.preciousedu.in"

    def test_enforces_https(self, normalizer):
        result = normalizer.normalize("http://www.preciousedu.in/about")
        assert result is not None
        assert result.startswith("https://")
