"""
Precious Edu LLM — Integration Tests for Structured Q&A Knowledge Retrieval

Verifies actual semantic answer relevance for all 10 required prompt scenarios:
1. "help me to get the student visa" -> Student visa structured knowledge
2. "Are you providing IELTS classes?" -> IELTS classes/training knowledge
3. "in which time?" (follow-up after IELTS) -> IELTS duration/timing
4. "services" -> Structured services catalog
5. "what documents are required?" (follow-up after student visa) -> Admission/visa documents
6. "what does PEIC mean?" -> Full form: Precious Education and Immigration Consultant
7. Country-specific visa query -> Australian study visa knowledge
8. Unknown question -> Graceful fallback
9. Explicit privacy policy question -> Privacy policy knowledge
10. Explicit terms question -> Terms knowledge
"""

import pytest


@pytest.mark.asyncio
async def test_health_structured_qa(async_client):
    """Verifies the GET /api/health/structured-qa endpoint reports ready status."""
    res = await async_client.get("/api/health/structured-qa")
    assert res.status_code == 200
    data = res.json()
    assert data["is_ready"] is True
    assert data["total_active_records"] > 20000
    assert data["unique_intents_count"] > 100


@pytest.mark.asyncio
async def test_student_visa_acceptance(async_client):
    """
    Mandatory Acceptance Test:
    User: 'help me to get the student visa'
    Expected:
    - Retreived from structured_qa
    - Discusses student visa / study permit process
    - NEVER answers with generic company introduction
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "Student Visa Test"})
    assert sess_res.status_code == 201
    session_id = sess_res.json()["session_id"]

    chat_res = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "help me to get the student visa"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    answer = data["response"]
    metadata = data.get("metadata", {})

    # Must be from structured_qa
    assert metadata.get("response_source") == "structured_qa"

    # Must contain student visa / study permit content
    assert any(term in answer.lower() for term in ["student visa", "study permit", "study abroad", "sds"])

    # Must NOT be generic company background
    assert "precious education and immigration consultant was established in 2005" not in answer.lower()
    assert "management has total" not in answer.lower()


@pytest.mark.asyncio
async def test_ielts_and_timing_followup(async_client):
    """
    Test IELTS question followed by conversational follow-up:
    Turn 1: 'Are you providing IELTS classes?'
    Turn 2: 'in which time?'
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "IELTS Follow-up Test"})
    session_id = sess_res.json()["session_id"]

    # Turn 1: IELTS classes
    res1 = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "Are you providing IELTS classes?"
    })
    assert res1.status_code == 200
    ans1 = res1.json()["response"]
    meta1 = res1.json().get("metadata", {})
    assert meta1.get("response_source") == "structured_qa"
    assert any(term in ans1.lower() for term in ["ielts", "training", "coaching", "band"])
    assert "privacy policy" not in ans1.lower()

    # Turn 2: Follow-up 'in which time?'
    res2 = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "in which time?"
    })
    assert res2.status_code == 200
    ans2 = res2.json()["response"]
    meta2 = res2.json().get("metadata", {})
    assert meta2.get("response_source") == "structured_qa"
    # Should resolve to IELTS duration/time
    assert any(term in ans2.lower() for term in ["duration", "2 hours", "45 minutes", "time commitment"])


@pytest.mark.asyncio
async def test_services_question(async_client):
    """
    User: 'services'
    Expected:
    - Retreives structured service catalog
    - Mentions express entry / study visa / immigration visa services
    - Never returns Privacy Policy or Terms & Conditions
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "Services Test"})
    session_id = sess_res.json()["session_id"]

    chat_res = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "services"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    answer = data["response"]
    metadata = data.get("metadata", {})

    assert metadata.get("response_source") == "structured_qa"
    assert any(term in answer.lower() for term in ["immigration visa services", "express entry", "family sponsorship", "study visa", "services"])
    assert "privacy policy" not in answer.lower()
    assert "third-party social media service" not in answer.lower()


@pytest.mark.asyncio
async def test_student_visa_to_documents_followup(async_client):
    """
    Turn 1: 'help me to get the student visa'
    Turn 2: 'what documents are required?'
    Expected:
    - Turn 2 resolves to admission / study visa documents
    - Mentions transcripts / LORs / SOP / passport
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "Docs Follow-up Test"})
    session_id = sess_res.json()["session_id"]

    await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "help me to get the student visa"
    })

    res2 = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "what documents are required?"
    })
    assert res2.status_code == 200
    ans2 = res2.json()["response"]
    meta2 = res2.json().get("metadata", {})

    assert meta2.get("response_source") == "structured_qa"
    assert any(term in ans2.lower() for term in ["transcripts", "lors", "sop", "passport", "academic transcripts"])


@pytest.mark.asyncio
async def test_peic_fullform(async_client):
    """
    User: 'what does PEIC mean?'
    Expected:
    - Answers with 'Precious Education and Immigration Consultant'
    - Source is structured_qa (FullForm.json)
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "PEIC Test"})
    session_id = sess_res.json()["session_id"]

    chat_res = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "what does PEIC mean?"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    answer = data["response"]
    metadata = data.get("metadata", {})

    assert metadata.get("response_source") == "structured_qa"
    assert "precious education and immigration consultant" in answer.lower()


@pytest.mark.asyncio
async def test_country_specific_visa(async_client):
    """
    User: 'can students apply for australian study visas via peic?'
    Expected:
    - Answers with Australian student visa details (subclass 500)
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "Australia Visa Test"})
    session_id = sess_res.json()["session_id"]

    chat_res = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "can students apply for australian study visas via peic?"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    answer = data["response"]
    metadata = data.get("metadata", {})

    assert metadata.get("response_source") == "structured_qa"
    assert any(term in answer.lower() for term in ["subclass 500", "australia", "dependent", "student visa"])


@pytest.mark.asyncio
async def test_unknown_question_fallback(async_client):
    """
    User: 'What is the weather on Mars?'
    Expected:
    - Triggers fallback because no knowledge matches
    - Response source is fallback
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "Fallback Test"})
    session_id = sess_res.json()["session_id"]

    chat_res = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "What is the weather on Mars?"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    answer = data["response"]
    metadata = data.get("metadata", {})

    assert metadata.get("response_source") == "fallback"
    assert "knowledge base" in answer.lower()


@pytest.mark.asyncio
async def test_explicit_privacy_policy(async_client):
    """
    User: 'What is your privacy policy?'
    Expected:
    - Privacy Policy is retrievable when explicitly asked
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "Privacy Test"})
    session_id = sess_res.json()["session_id"]

    chat_res = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "What is your privacy policy?"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    answer = data["response"]

    assert any(term in answer.lower() for term in ["privacy policy", "privacy", "personal data"])


@pytest.mark.asyncio
async def test_explicit_terms_and_conditions(async_client):
    """
    User: 'what location does the term country refer to under these terms?'
    Expected:
    - Terms and Conditions is retrievable when explicitly asked
    """
    sess_res = await async_client.post("/api/sessions", json={"title": "Terms Test"})
    session_id = sess_res.json()["session_id"]

    chat_res = await async_client.post("/api/chat", json={
        "session_id": session_id,
        "message": "what location does the term country refer to under these terms?"
    })
    assert chat_res.status_code == 200
    data = chat_res.json()
    answer = data["response"]

    assert any(term in answer.lower() for term in ["terms", "service", "country", "individual accessing"])
