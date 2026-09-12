"""
Unit Tests — URLDiscoverer

Covers: anchor tag extraction, sitemap parsing, domain restriction,
malformed HTML handling, and mailto/javascript filtering.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.website.crawler.url_discovery import URLDiscoverer
from app.website.crawler.url_normalizer import URLNormalizer


@pytest.fixture
def discoverer():
    return URLDiscoverer(normalizer=URLNormalizer())


BASE_URL = "https://www.preciousedu.in/"


class TestDiscoverFromHtml:
    def test_extracts_internal_links(self, discoverer):
        html = """
        <html><body>
          <a href="/services">Services</a>
          <a href="/contact">Contact</a>
        </body></html>
        """
        links = discoverer.discover_from_html(html, BASE_URL)
        assert any("services" in l for l in links)
        assert any("contact" in l for l in links)

    def test_filters_external_links(self, discoverer):
        html = '<a href="https://google.com/search">Google</a>'
        links = discoverer.discover_from_html(html, BASE_URL)
        assert all("google.com" not in l for l in links)

    def test_filters_mailto_links(self, discoverer):
        html = '<a href="mailto:info@preciousedu.in">Email</a>'
        links = discoverer.discover_from_html(html, BASE_URL)
        assert links == []

    def test_filters_javascript_links(self, discoverer):
        html = '<a href="javascript:void(0)">Click</a>'
        links = discoverer.discover_from_html(html, BASE_URL)
        assert links == []

    def test_filters_anchor_only_links(self, discoverer):
        html = '<a href="#section-top">Top</a>'
        links = discoverer.discover_from_html(html, BASE_URL)
        assert links == []

    def test_deduplicates_links(self, discoverer):
        html = """
        <a href="/about">About</a>
        <a href="/about">About Again</a>
        <a href="/about">About Third</a>
        """
        links = discoverer.discover_from_html(html, BASE_URL)
        about_links = [l for l in links if "about" in l]
        assert len(about_links) == 1

    def test_resolves_relative_paths(self, discoverer):
        html = '<a href="visa/student">Visa</a>'
        links = discoverer.discover_from_html(html, BASE_URL)
        assert any("visa" in l for l in links)

    def test_empty_html_returns_empty(self, discoverer):
        assert discoverer.discover_from_html("", BASE_URL) == []

    def test_malformed_html_handled_gracefully(self, discoverer):
        html = "<a href='/services'>Unclosed <b>tag"
        links = discoverer.discover_from_html(html, BASE_URL)
        assert isinstance(links, list)


@pytest.mark.asyncio
class TestDiscoverFromSitemap:
    async def test_extracts_sitemap_urls(self, discoverer):
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://www.preciousedu.in/</loc></url>
          <url><loc>https://www.preciousedu.in/services</loc></url>
          <url><loc>https://www.preciousedu.in/contact</loc></url>
        </urlset>"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sitemap_xml

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            urls = await discoverer.discover_from_sitemap(BASE_URL)

        assert len(urls) >= 2
        assert any("services" in u for u in urls)

    async def test_missing_sitemap_returns_empty(self, discoverer):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = ""

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            urls = await discoverer.discover_from_sitemap(BASE_URL)

        assert urls == []
