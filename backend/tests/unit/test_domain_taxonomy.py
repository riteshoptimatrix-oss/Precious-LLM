import pytest
from app.domain.taxonomy import DomainTaxonomy, DomainCategory

def test_domain_taxonomy_defaults():
    taxonomy = DomainTaxonomy.get_default_taxonomy()
    assert "study_visa" in taxonomy.categories
    assert "immigration" in taxonomy.categories
    assert "test_prep" in taxonomy.categories

def test_domain_taxonomy_lookup():
    taxonomy = DomainTaxonomy.get_default_taxonomy()
    cat = taxonomy.categories.get("study_visa")
    assert cat is not None
    assert cat.name == "Study Visa Services"

def test_domain_taxonomy_prohibited_assumptions():
    taxonomy = DomainTaxonomy.get_default_taxonomy()
    assumptions = taxonomy.prohibited_assumptions
    assert any("project" in a.lower() for a in assumptions)

def test_domain_taxonomy_serialization(tmp_path):
    taxonomy = DomainTaxonomy.get_default_taxonomy()
    file_path = tmp_path / "taxonomy.json"
    taxonomy.save_json(file_path)
    
    loaded = DomainTaxonomy.load_json(file_path)
    assert loaded.domain_name == taxonomy.domain_name
    assert "study_visa" in loaded.categories
