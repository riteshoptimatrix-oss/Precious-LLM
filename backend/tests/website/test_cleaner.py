import pytest
from app.website.extraction.cleaner import HTMLCleaner

def test_cleaner_removes_unwanted_tags():
    html = """
    <html>
    <head><style>body { color: red; }</style></head>
    <body>
        <nav><a href="#">Home</a></nav>
        <header><h1>Site Header</h1></header>
        <main>
            <p>Main content text here.</p>
            <script>console.log('test');</script>
        </main>
        <footer>Footer text</footer>
    </body>
    </html>
    """
    clean_text = HTMLCleaner.clean(html)
    assert "body { color: red; }" not in clean_text
    assert "console.log" not in clean_text
    assert "Site Header" not in clean_text
    assert "Footer text" not in clean_text
    assert "Main content text here." in clean_text

def test_cleaner_unicode_preservation():
    html = "<p>Precious Education coaching guidance in Hindi/English: उच्च शिक्षा मार्गदर्शन</p>"
    clean_text = HTMLCleaner.clean(html)
    assert "उच्च शिक्षा मार्गदर्शन" in clean_text
