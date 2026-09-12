import pytest
from app.website.extraction.boilerplate import BoilerplateFilter

def test_boilerplate_filter_default():
    bp = BoilerplateFilter()
    text = "Study Abroad Services\nHome\nContact Us\nWe offer student visa assistance."
    filtered = bp.filter_text(text)
    
    assert "Study Abroad Services" in filtered
    assert "We offer student visa assistance." in filtered
    assert "Home" not in filtered
    assert "Contact Us" not in filtered

def test_boilerplate_filter_custom():
    bp = BoilerplateFilter(boilerplate_lines={"custom footer text"})
    text = "Important content\ncustom footer text\nMore content"
    filtered = bp.filter_text(text)
    
    assert "Important content" in filtered
    assert "More content" in filtered
    assert "custom footer text" not in filtered
