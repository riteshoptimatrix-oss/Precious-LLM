"""
Precious AI — HTML Cleaner & Unicode Text Normalizer

Strips script, style, nav, footer, header, cookie banners, tracking code,
and preserves UTF-8 / Hindi text.
"""

import re
from bs4 import BeautifulSoup, Comment


class HTMLCleaner:
    """
    Cleans raw HTML and converts to safe, readable text.
    """

    UNWANTED_TAGS = {"script", "style", "nav", "header", "footer", "iframe", "noscript", "svg", "form", "button"}

    @classmethod
    def clean(cls, html_text: str) -> str:
        """
        Removes non-content elements and extracts clean plain text.
        """
        if not html_text:
            return ""

        soup = BeautifulSoup(html_text, "html.parser")

        # 1. Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 2. Remove unwanted tags
        for tag_name in cls.UNWANTED_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 3. Extract text
        text = soup.get_text(separator="\n")

        # 4. Clean line breaks and spacing
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines = [line for line in lines if line]

        return "\n".join(cleaned_lines)
