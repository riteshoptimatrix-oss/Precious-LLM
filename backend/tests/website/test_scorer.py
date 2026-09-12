import pytest
from app.website.search.scorer import RelevanceScorer

def test_scorer_title_match():
    chunk = {
        "title": "F1 Student Visa Guidance",
        "section": "Overview",
        "content": "Step by step process to get your student visa."
    }
    score = RelevanceScorer.score_chunk(chunk, "F1 Student Visa", ["f1", "student", "visa"])
    assert score > 50.0

def test_scorer_no_match():
    chunk = {
        "title": "Cooking Recipes",
        "section": "Italian Food",
        "content": "Pasta and pizza preparation instructions."
    }
    score = RelevanceScorer.score_chunk(chunk, "F1 Student Visa", ["f1", "student", "visa"])
    assert score == 0.0
