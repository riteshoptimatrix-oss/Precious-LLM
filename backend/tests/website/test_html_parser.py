import pytest
from app.website.extraction.html_parser import HTMLDocumentParser

def test_html_document_parser_basic():
    parser = HTMLDocumentParser()
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Precious Education - Study Visa Assistance</title>
    </head>
    <body>
        <h1>Study Abroad Services</h1>
        <p>We provide full visa counseling for USA, UK, and Canada.</p>
        <h2>Countries Covered</h2>
        <p>USA, UK, Canada, Australia, and Germany.</p>
    </body>
    </html>
    """
    url = "https://www.preciousedu.in/services"
    parsed = parser.parse(html, url)

    assert parsed["url"] == url
    assert parsed["title"] == "Precious Education - Study Visa Assistance"
    assert "Study Abroad Services" in parsed["headings"]
    assert "Countries Covered" in parsed["headings"]
    assert parsed["page_type"] in ("service", "study", "country")
    assert len(parsed["sections"]) >= 1

def test_html_document_parser_empty():
    parser = HTMLDocumentParser()
    parsed = parser.parse("", "https://www.preciousedu.in/")
    assert parsed["title"] == ""
    assert parsed["headings"] == []
    assert parsed["sections"] == []
