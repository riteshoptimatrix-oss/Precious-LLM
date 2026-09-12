"""
Precious AI — Link Discovery Module

Extracts candidate URLs from HTML markup (`<a href="...">`) and sitemap.xml.
Normalizes discovered links and checks domain boundaries.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Set, Optional
from bs4 import BeautifulSoup
import httpx

from app.website.crawler.url_normalizer import URLNormalizer


class URLDiscoverer:
    """
    Discovers new links from HTML documents and XML sitemaps.
    """

    def __init__(self, normalizer: Optional[URLNormalizer] = None):
        self.normalizer = normalizer or URLNormalizer()

    def discover_from_html(self, html_text: str, base_url: str) -> List[str]:
        """
        Parses HTML content, finds anchor tags (`<a href="...">`), and returns normalized URLs.
        """
        if not html_text:
            return []

        discovered = []
        seen = set()

        try:
            soup = BeautifulSoup(html_text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "").strip()
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue

                norm_url = self.normalizer.normalize(href, base_url=base_url)
                if norm_url and norm_url not in seen:
                    seen.add(norm_url)
                    discovered.append(norm_url)
        except Exception as e:
            pass

        return discovered

    async def discover_from_sitemap(self, base_url: str) -> List[str]:
        """
        Fetches /sitemap.xml from domain root if available and extracts declared page URLs.
        """
        sitemap_url = f"{base_url.rstrip('/')}/sitemap.xml"
        discovered = []
        seen = set()

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(sitemap_url)
                if resp.status_code == 200 and resp.text:
                    # Parse XML
                    root = ET.fromstring(resp.text)
                    for elem in root.iter():
                        if elem.tag.endswith("loc") and elem.text:
                            loc_url = elem.text.strip()
                            norm_url = self.normalizer.normalize(loc_url, base_url=base_url)
                            if norm_url and norm_url not in seen:
                                seen.add(norm_url)
                                discovered.append(norm_url)
        except Exception:
            pass

        return discovered
