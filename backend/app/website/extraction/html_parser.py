"""
Precious AI — Semantic HTML Document Parser

Extracts page titles, heading structures, section hierarchies, main text, and tables.
"""

from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

from app.website.extraction.cleaner import HTMLCleaner
from app.website.extraction.boilerplate import BoilerplateFilter
from app.website.extraction.classifier import PageClassifier


class HTMLDocumentParser:
    """
    Parses HTML documents into structured semantic sections.
    """

    def __init__(self, boilerplate_filter: Optional[BoilerplateFilter] = None):
        self.bp_filter = boilerplate_filter or BoilerplateFilter()

    def parse(self, html_text: str, url: str) -> Dict[str, Any]:
        """
        Parses raw HTML into structured page metadata and content sections.
        """
        if not html_text:
            return {
                "url": url,
                "title": "",
                "headings": [],
                "page_type": "other",
                "clean_content": "",
                "sections": []
            }

        soup = BeautifulSoup(html_text, "html.parser")

        # 1. Extract Title
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else ""
        if not title:
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()

        # 2. Extract Headings (h1, h2, h3)
        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            text = h.get_text().strip()
            if text:
                headings.append(text)

        # 3. Classify Page Type
        page_type = PageClassifier.classify(url, title, headings)

        # 4. Extract Main Content Body
        main_tag = soup.find("main") or soup.find("article") or soup.find("body") or soup
        raw_clean_text = HTMLCleaner.clean(str(main_tag))
        filtered_content = self.bp_filter.filter_text(raw_clean_text)

        # 5. Extract Section Hierarchy
        sections = []
        current_heading = title or "Overview"
        current_level = 1
        current_paras = []

        for elem in soup.find_all(["h1", "h2", "h3", "p"]):
            name = elem.name
            text = elem.get_text().strip()
            if not text:
                continue

            if name in ("h1", "h2", "h3"):
                if current_paras:
                    sec_text = self.bp_filter.filter_text("\n".join(current_paras))
                    if sec_text:
                        sections.append({
                            "heading": current_heading,
                            "level": current_level,
                            "content": sec_text
                        })
                    current_paras = []
                current_heading = text
                current_level = int(name[1])
            else:
                current_paras.append(text)

        if current_paras:
            sec_text = self.bp_filter.filter_text("\n".join(current_paras))
            if sec_text:
                sections.append({
                    "heading": current_heading,
                    "level": current_level,
                    "content": sec_text
                })

        return {
            "url": url,
            "title": title or "Precious Education",
            "headings": headings,
            "page_type": page_type,
            "clean_content": filtered_content,
            "sections": sections if sections else [{
                "heading": title or "Overview",
                "level": 1,
                "content": filtered_content
            }]
        }
