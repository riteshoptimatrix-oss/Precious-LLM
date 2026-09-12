"""
Precious AI — Rule-Based Page Classifier

Classifies webpage type using URL path, title, and headings without external AI models.
"""

import urllib.parse
from typing import List, Dict


class PageClassifier:
    """
    Deterministic webpage category classifier.
    """

    CATEGORIES = {
        "home": ["home", "index"],
        "about": ["about", "about-us", "who-we-are", "company"],
        "service": ["service", "services", "coaching", "assistance", "counseling"],
        "visa": ["visa", "student-visa", "work-visa", "tourist-visa", "f1", "m1"],
        "country": ["country", "usa", "uk", "canada", "australia", "germany", "study-abroad"],
        "study": ["study", "course", "university", "college", "admission"],
        "immigration": ["immigration", "pr", "permanent-residency"],
        "ielts": ["ielts", "toefl", "pte", "gre", "gmat", "test-prep"],
        "contact": ["contact", "contact-us", "reach-us", "location", "address"],
        "blog": ["blog", "news", "article", "updates"],
        "faq": ["faq", "frequently-asked-questions", "questions"],
    }

    @classmethod
    def classify(cls, url: str, title: str, headings: List[str]) -> str:
        """
        Classifies page into a canonical type string.
        """
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()
        title_lower = title.lower() if title else ""
        headings_text = " ".join(headings).lower() if headings else ""

        # Root homepage check
        if path in ("", "/", "/index.html", "/home"):
            return "home"

        # Check each category against path, title, headings
        for category, keywords in cls.CATEGORIES.items():
            if category == "home":
                continue
            for kw in keywords:
                if f"/{kw}" in path or f"-{kw}" in path or kw in title_lower or kw in headings_text:
                    return category

        return "other"
