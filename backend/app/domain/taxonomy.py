"""
Precious Edu LLM — Domain Taxonomy Module

Defines domain categories, subcategories, terminology mappings, synonyms,
common intents, prohibited assumptions, and response guidelines for Precious Education.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DomainCategory(BaseModel):
    """Domain category definition."""
    name: str
    description: str
    subcategories: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)


class DomainTaxonomy(BaseModel):
    """Domain taxonomy schema for Precious Education."""
    domain_name: str = "Precious Education Consultancy"
    version: str = "1.0.0"
    categories: Dict[str, DomainCategory] = Field(default_factory=dict)
    terminology_map: Dict[str, str] = Field(default_factory=dict)
    prohibited_assumptions: List[str] = Field(default_factory=list)
    response_guidelines: List[str] = Field(default_factory=list)

    @classmethod
    def get_default_taxonomy(cls) -> "DomainTaxonomy":
        """Generates standard Precious Education consultancy taxonomy."""
        return cls(
            domain_name="Precious Education Consultancy",
            version="1.0.0",
            categories={
                "study_visa": DomainCategory(
                    name="Study Visa Services",
                    description="Assistance with F-1, student visas, university admissions, and student guidance.",
                    subcategories=["F1 Visa", "M1 Visa", "University Choice", "I-20 Form"],
                    synonyms=["student visa", "study permit", "education visa", "f1", "f-1"]
                ),
                "immigration": DomainCategory(
                    name="Immigration & Permanent Residency",
                    description="Consultancy for skilled worker immigration, PR pathways, and legal entry.",
                    subcategories=["Skilled Migration", "PR Visa", "Express Entry", "Points Calculation"],
                    synonyms=["pr", "permanent residency", "green card", "immigration service"]
                ),
                "visitor_visa": DomainCategory(
                    name="Visitor & Tourist Visa",
                    description="Short-term tourist, business, and family visit visas.",
                    subcategories=["B1/B2 Visa", "Tourist Visa", "Family Sponsor"],
                    synonyms=["tourist visa", "visitor visa", "b1/b2", "travel visa"]
                ),
                "work_permit": DomainCategory(
                    name="Work Permits",
                    description="Employment visas, post-study work permits (OPT/CPT), and work authorization.",
                    subcategories=["OPT", "CPT", "H-1B", "Work Visa"],
                    synonyms=["work visa", "work permit", "opt", "cpt", "employment visa"]
                ),
                "test_prep": DomainCategory(
                    name="Language & Test Preparation",
                    description="Preparation for IELTS, TOEFL, GRE, GMAT, and PTE examinations.",
                    subcategories=["IELTS", "TOEFL", "PTE", "GRE", "GMAT"],
                    synonyms=["ielts", "toefl", "pte", "gre", "test prep", "english exam"]
                )
            },
            terminology_map={
                "f1": "F1 Academic Student Visa",
                "f-1": "F1 Academic Student Visa",
                "m1": "M1 Vocational Student Visa",
                "m-1": "M1 Vocational Student Visa",
                "opt": "Optional Practical Training (OPT)",
                "cpt": "Curricular Practical Training (CPT)",
                "i20": "Form I-20 (Certificate of Eligibility for Nonimmigrant Student Status)",
                "i-20": "Form I-20 (Certificate of Eligibility for Nonimmigrant Student Status)",
                "ielts": "International English Language Testing System (IELTS)",
                "toefl": "Test of English as a Foreign Language (TOEFL)"
            },
            prohibited_assumptions=[
                "Do NOT invent or fabricate live project statuses. Direct project queries to Knowledge Engine.",
                "Do NOT state fixed embassy fees without mentioning they are subject to official government updates.",
                "Do NOT guarantee 100% visa approval."
            ],
            response_guidelines=[
                "Be professional, encouraging, clear, and concise.",
                "Distinguish general visa rules from country-specific procedures.",
                "Ask clarifying questions when user intent or target country is unspecified.",
                "Preserve multi-turn conversational context."
            ]
        )

    def save_json(self, path: str | Path) -> None:
        """Save taxonomy to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "DomainTaxonomy":
        """Load taxonomy from JSON file."""
        p = Path(path)
        if not p.exists():
            default_tax = cls.get_default_taxonomy()
            default_tax.save_json(p)
            return default_tax
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
