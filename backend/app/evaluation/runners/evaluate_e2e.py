"""
Precious AI — Level 5 Full End-to-End & Session Isolation Evaluator Runner

Evaluates session isolation (Session A vs Session B memory leakage), multi-step complete realistic conversation flows,
and full HTTP API flow (PHP/FastAPI → Chat Service → Conversation Engine → Knowledge Engine → Custom LLM → MongoDB).
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.evaluation.runners.runner_base import EvaluationRunnerBase
from app.evaluation.metrics.eval_metrics import EvaluationMetrics
from app.knowledge.models import ProjectRecord, SourceMetadata
from app.knowledge.repository import ProjectRepository


class EndToEndEvaluator(EvaluationRunnerBase):
    """
    Evaluator for End-to-End System Flow and Session Isolation.
    """

    def __init__(self):
        super().__init__(db_name="precious_ai_eval_e2e")

    async def seed_project_data(self) -> None:
        """Seeds project records for E2E testing."""
        repo = ProjectRepository(self.db)
        records = [
            ProjectRecord(
                project_id="P001",
                project_name="ABC",
                client="Acme Corp",
                status="In Progress",
                manager="John",
                source=SourceMetadata(
                    file="e2e_projects.xlsx",
                    sheet="Projects",
                    row=2,
                    file_hash="hash_e2e_eval",
                    dataset_version="projects-e2e-v1"
                ),
                dataset_version="projects-e2e-v1"
            )
        ]
        await repo.save_dataset(
            records=records,
            dataset_version="projects-e2e-v1",
            file_hash="hash_e2e_eval",
            source_file="e2e_projects.xlsx",
            activate=True
        )

    async def run(self) -> Dict[str, Any]:
        """
        Executes Session Isolation and Full Realistic E2E Conversation Scenarios.
        """
        await self.setup()
        await self.seed_project_data()

        results = []
        start_time = time.perf_counter()

        # ---------------------------------------------------------
        # 1. Session Isolation Test (Session A vs Session B)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        session_a = await self.session_service.create_session("Session A")
        session_b = await self.session_service.create_session("Session B")

        session_a_id = session_a["session_id"]
        session_b_id = session_b["session_id"]

        # Session A: My name is Ritesh.
        res_a1 = await self.engine.handle_message(session_a_id, "My name is Ritesh.")

        # Session B: What is my name?
        res_b1 = await self.engine.handle_message(session_b_id, "What is my name?")
        resp_b1_text = res_b1.get("response", "")

        # Session B MUST NOT leak Ritesh from Session A
        has_leak = "ritesh" in resp_b1_text.lower()
        passed_isolation = not has_leak
        lat_iso = time.perf_counter() - t0

        results.append({
            "id": "e2e_session_isolation",
            "category": "session_isolation",
            "prompt": "What is my name? (Session B without prior introduction)",
            "response": resp_b1_text,
            "passed": passed_isolation,
            "latency": lat_iso,
            "failure_category": None if passed_isolation else "MEMORY"
        })

        # ---------------------------------------------------------
        # 2. Multi-step End-to-End Realistic Flow Test
        # ---------------------------------------------------------
        conversation_flow = [
            ("Hello", "greeting"),
            ("My name is Ritesh.", "name_intro"),
            ("I want to study in the USA.", "intent_intro"),
            ("Which visa would I need?", "domain_visa"),
            ("What is the difference between F1 and M1?", "domain_comparison"),
            ("Now tell me about Project ABC.", "project_intro"),
            ("What is its current status?", "project_status_contextual")
        ]

        session_e2e = await self.session_service.create_session("Full E2E Scenario Session")
        e2e_id = session_e2e["session_id"]

        flow_passed = True
        flow_details = []

        for prompt, tag in conversation_flow:
            t_step = time.perf_counter()
            try:
                res = await self.engine.handle_message(e2e_id, prompt)
                lat_step = time.perf_counter() - t_step
                resp = res.get("response", "")
                valid = EvaluationMetrics.is_valid_response(resp)

                if not valid:
                    flow_passed = False

                flow_details.append({
                    "step": tag,
                    "prompt": prompt,
                    "response": resp,
                    "passed": valid,
                    "latency": lat_step
                })
            except Exception as e:
                flow_passed = False
                flow_details.append({
                    "step": tag,
                    "prompt": prompt,
                    "error": str(e),
                    "passed": False
                })

        results.append({
            "id": "e2e_full_conversation_flow",
            "category": "full_e2e_flow",
            "passed": flow_passed,
            "steps": flow_details,
            "failure_category": None if flow_passed else "CONTEXT"
        })

        # ---------------------------------------------------------
        # 3. HTTP API End-to-End Integration Check
        # ---------------------------------------------------------
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create session via HTTP
            s_resp = await client.post("/api/sessions", json={"title": "HTTP E2E Test Session"})
            if s_resp.status_code in (200, 201):
                http_session_id = s_resp.json()["session_id"]
                c_resp = await client.post("/api/chat", json={"session_id": http_session_id, "message": "Hello"})
                api_passed = c_resp.status_code == 200 and "response" in c_resp.json()
            else:
                api_passed = False

            results.append({
                "id": "e2e_http_fastapi_flow",
                "category": "api_integration",
                "passed": api_passed,
                "failure_category": None if api_passed else "API"
            })

        total_time = time.perf_counter() - start_time
        await self.teardown()

        summary = EvaluationMetrics.compute_summary_scorecard(results)
        summary["total_duration_sec"] = total_time
        summary["results"] = results
        return summary


def main():
    evaluator = EndToEndEvaluator()
    summary = asyncio.run(evaluator.run())
    print("\n========================================")
    print("END-TO-END SYSTEM EVALUATION SUMMARY")
    print("========================================")
    print(f"Total Tests : {summary['total_tests']}")
    print(f"Passed      : {summary['passed']}")
    print(f"Failed      : {summary['failed']}")
    print(f"Pass Rate   : {summary['pass_rate']*100:.1f}%")
    print(f"Duration    : {summary['total_duration_sec']:.2f}s")
    print("========================================\n")


if __name__ == "__main__":
    main()
