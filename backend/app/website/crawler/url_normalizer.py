"""
Precious AI — URL Normalizer & Domain Restriction Checker

Normalizes relative & absolute URLs, strips fragments, sorts query parameters,
removes tracking params, and enforces domain boundaries.
"""

import urllib.parse
from typing import List, Optional, Set


class URLNormalizer:
    """
    URL Normalization & Security Checker.
    """

    ALLOWED_DOMAINS = {"preciousedu.in", "www.preciousedu.in"}
    STRIP_QUERY_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid", "gclid"}

    def __init__(self, allowed_domains: Optional[Set[str]] = None, enforce_https: bool = True):
        self.allowed_domains = allowed_domains or self.ALLOWED_DOMAINS
        self.enforce_https = enforce_https

    def is_allowed_domain(self, url: str) -> bool:
        """
        Verifies that the target URL belongs to an allowed domain.
        Prevents SSRF and unauthorized domain crawling.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.netloc:
                return False
            hostname = parsed.netloc.split(":")[0].lower()
            return hostname in self.allowed_domains
        except Exception:
            return False

    def normalize(self, raw_url: str, base_url: Optional[str] = None) -> Optional[str]:
        """
        Normalizes a raw or relative URL into a canonical URL string.
        - Resolves relative paths against base_url
        - Strips fragments (#section)
        - Lowercases scheme and hostname
        - Normalizes trailing slash
        - Strips tracking parameters
        """
        if not raw_url:
            return None

        # Resolve relative URL
        if base_url:
            raw_url = urllib.parse.urljoin(base_url, raw_url)

        try:
            parsed = urllib.parse.urlparse(raw_url)
            if parsed.scheme not in ("http", "https"):
                return None

            scheme = "https" if self.enforce_https else parsed.scheme.lower()
            hostname = parsed.netloc.split(":")[0].lower()

            if hostname not in self.allowed_domains:
                return None

            path = parsed.path
            # Normalize path slashes
            if not path:
                path = "/"
            elif path != "/" and path.endswith("/"):
                path = path[:-1]

            # Filter & sort query parameters
            query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
            filtered_query = [
                (k, v) for k, v in query_pairs if k.lower() not in self.STRIP_QUERY_PARAMS
            ]
            filtered_query.sort(key=lambda x: x[0])
            sorted_query = urllib.parse.urlencode(filtered_query)

            # Reconstruct normalized URL
            clean_url = urllib.parse.urlunparse((
                scheme,
                hostname,
                path,
                "",  # params
                sorted_query,
                ""   # fragment stripped
            ))
            return clean_url
        except Exception:
            return None
