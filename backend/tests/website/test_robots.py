"""
Unit Tests — RobotsChecker

Covers: allow/disallow parsing, default-allow on fetch failure,
and domain-scoped caching.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.website.crawler.robots import RobotsChecker


@pytest.fixture
def checker():
    return RobotsChecker(user_agent="PreciousAICrawler/1.0")


class TestIsAllowed:
    def test_allows_all_when_no_robots_fetched(self, checker):
        """Without fetching, default is allow-all."""
        assert checker.is_allowed("https://www.preciousedu.in/about") is True

    def test_disallowed_path_blocked(self, checker):
        """Manually inject a parser that blocks /admin."""
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /admin/"])
        checker.parsers["www.preciousedu.in"] = rp

        assert checker.is_allowed("https://www.preciousedu.in/admin/secret") is False

    def test_allowed_path_passes(self, checker):
        """Paths not in Disallow are permitted."""
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /admin/"])
        checker.parsers["www.preciousedu.in"] = rp

        assert checker.is_allowed("https://www.preciousedu.in/services") is True

    def test_empty_robots_allows_all(self, checker):
        """Empty robots.txt rules → allow everything."""
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        rp.parse([])
        checker.parsers["www.preciousedu.in"] = rp

        assert checker.is_allowed("https://www.preciousedu.in/anything") is True


@pytest.mark.asyncio
class TestFetchAndParse:
    async def test_fetch_parses_rules(self, checker):
        """fetch_and_parse populates parsers dict for domain."""
        robots_text = "User-agent: *\nDisallow: /private/\n"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = robots_text

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_resp)
            MockClient.return_value = instance

            await checker.fetch_and_parse("https://www.preciousedu.in/")

        assert "www.preciousedu.in" in checker.parsers

    async def test_fetch_failure_defaults_allow_all(self, checker):
        """On network error, allow-all is the safe default."""
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=ConnectionError("timeout"))
            MockClient.return_value = instance

            await checker.fetch_and_parse("https://www.preciousedu.in/")

        assert checker.is_allowed("https://www.preciousedu.in/anything") is True

    async def test_caches_result_on_second_call(self, checker):
        """Second call for same domain should not re-fetch."""
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        rp.parse([])
        checker.parsers["www.preciousedu.in"] = rp  # Pre-populate

        with patch("httpx.AsyncClient") as MockClient:
            MockClient.assert_not_called()
            await checker.fetch_and_parse("https://www.preciousedu.in/")
            MockClient.assert_not_called()
