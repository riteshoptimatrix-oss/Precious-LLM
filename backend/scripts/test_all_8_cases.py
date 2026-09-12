import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database
from app.conversation.engine import ConversationEngine
from app.llm import get_llm_service
from app.repositories.session_repository import SessionRepository
from app.models.session import SessionModel

async def run_tests():
    await connect_to_mongodb()
    db = await get_database()
    llm_svc = get_llm_service()
    llm_svc.initialize()

    engine = ConversationEngine(db=db, generator=llm_svc)
    session_repo = SessionRepository(db)

    # Test Session
    session_id = f"test-all-8-{uuid.uuid4().hex[:8]}"
    await session_repo.create(SessionModel(session_id=session_id, user_id="test-user"))

    test_cases = [
        ("TEST 1", "Are you providing IELTS classes?"),
        ("TEST 3", "services"),
        ("TEST 4", "help me to get the student visa"),
        ("TEST 5", "I need help with a study visa"),
        ("TEST 6", "I want to study abroad"),
        ("TEST 7", "What countries do you provide student visa services for?"),
        ("TEST 8", "What is the mass of Saturn's moons?"),
    ]

    for label, msg in test_cases:
        print("="*70)
        print(f"=== {label} ===")
        print(f"USER INPUT: '{msg}'")
        res = await engine.handle_message(session_id=session_id, user_message=msg)
        print(f"ASSISTANT RESPONSE:\n{res['response']}\n")
        print(f"SOURCES: {res.get('sources')}")
        print(f"METADATA: {res.get('metadata')}")

    # TEST 2 (Follow-up)
    session_id_2 = f"test-fu-{uuid.uuid4().hex[:8]}"
    await session_repo.create(SessionModel(session_id=session_id_2, user_id="test-user"))
    print("\n" + "="*70)
    print("=== TEST 2 (Multi-Turn Follow-Up Query) ===")
    print("Turn 1 USER: 'Are you providing IELTS classes?'")
    res1 = await engine.handle_message(session_id=session_id_2, user_message="Are you providing IELTS classes?")
    print(f"Turn 1 BOT:\n{res1['response']}\n")

    print("Turn 2 USER: 'in which time?'")
    res2 = await engine.handle_message(session_id=session_id_2, user_message="in which time?")
    print(f"Turn 2 BOT:\n{res2['response']}\n")
    print(f"Turn 2 SOURCES: {res2.get('sources')}")
    print(f"Turn 2 METADATA: {res2.get('metadata')}")

if __name__ == "__main__":
    asyncio.run(run_tests())
