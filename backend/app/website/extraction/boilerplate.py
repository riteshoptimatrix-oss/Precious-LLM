"""
Precious AI — Boilerplate Content Filter

Detects and removes repeated site-wide navigation/footer boilerplate phrases.
"""

from typing import List, Optional, Set, Dict


class BoilerplateFilter:
    """
    Identifies and strips repeating boilerplate blocks.
    """

    DEFAULT_BOILERPLATE = {
        "home", "about us", "services", "contact us", "all rights reserved",
        "copyright", "terms & conditions", "privacy policy", "follow us on"
    }

    def __init__(self, boilerplate_lines: Optional[Set[str]] = None):
        self.boilerplate_lines = set(l.lower() for l in (boilerplate_lines or self.DEFAULT_BOILERPLATE))

    def filter_text(self, text: str) -> str:
        """
        Filters out common boilerplate lines from clean text.
        """
        if not text:
            return ""

        lines = text.splitlines()
        filtered = []
        for line in lines:
            line_str = line.strip()
            if line_str.lower() in self.boilerplate_lines:
                continue
            filtered.append(line_str)

        return "\n".join(filtered)
