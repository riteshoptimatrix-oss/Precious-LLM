"""
Precious AI — Robots.txt Compliance Inspector

Fetches, parses, and caches robots.txt rules to ensure crawler compliance.
"""

import urllib.robotparser
import urllib.parse
import logging
from typing import Dict, Optional
import httpx

logger = logging.getLogger(__name__)


class RobotsChecker:
    """
    Manages robots.txt rules for allowed domains.
    """

    def __init__(self, user_agent: str = "PreciousAICrawler/1.0"):
        self.user_agent = user_agent
        self.parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}

    async def fetch_and_parse(self, base_url: str) -> None:
        """
        Fetches robots.txt from domain root and caches rules.
        """
        parsed = urllib.parse.urlparse(base_url)
        domain = parsed.netloc.lower()

        if domain in self.parsers:
            return

        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(robots_url)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                    logger.info(f"Successfully fetched robots.txt from {robots_url}")
                else:
                    rp.parse([])  # Allow all if robots.txt absent or 404
        except Exception as e:
            logger.warning(f"Could not fetch robots.txt from {robots_url}: {e}")
            rp.parse([])  # Default allow all on fetch failure

        self.parsers[domain] = rp

    def is_allowed(self, url: str) -> bool:
        """
        Returns True if URL is allowed according to robots.txt rules.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            rp = self.parsers.get(domain)

            if rp is None:
                return True  # If not fetched yet, allow default

            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True
