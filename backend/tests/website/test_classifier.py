import pytest
from app.website.extraction.classifier import PageClassifier

def test_page_classifier_home():
    cat = PageClassifier.classify("https://www.preciousedu.in/", "Home - Precious Education", ["Welcome"])
    assert cat == "home"

def test_page_classifier_visa():
    cat = PageClassifier.classify("https://www.preciousedu.in/student-visa", "F1 Student Visa Guidance", ["Visa Types", "F1 Visa"])
    assert cat == "visa"

def test_page_classifier_country():
    cat = PageClassifier.classify("https://www.preciousedu.in/study-in-usa", "Study in USA", ["Universities in USA"])
    assert cat == "country"

def test_page_classifier_other():
    cat = PageClassifier.classify("https://www.preciousedu.in/random-page", "Random Page", ["Info"])
    assert cat == "other"
