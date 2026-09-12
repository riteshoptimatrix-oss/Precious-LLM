"""
Precious Edu LLM — Knowledge Intent Router

Determines whether a user prompt requires project data, website data, or both.
Uses deterministic regex and rule matching (no secondary LLMs).
Translates project questions into StructuredProjectQuery objects.
Supports multi-turn pronoun resolution ("it", "that project").
"""

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.knowledge.models import StructuredProjectQuery

logger = logging.getLogger(__name__)


class IntentClassificationResult:
    """
    Structured result of intent classification including resolved multi-turn query.
    """
    def __init__(
        self,
        predicted_intent: str,
        confidence: float,
        resolved_query: str,
        previous_context: Optional[str] = None
    ):
        self.predicted_intent = predicted_intent
        self.confidence = confidence
        self.resolved_query = resolved_query
        self.previous_context = previous_context


class RoutingDecision(str, Enum):
    NO_KNOWLEDGE_REQUIRED = "NO_KNOWLEDGE_REQUIRED"
    PROJECT_KNOWLEDGE_REQUIRED = "PROJECT_KNOWLEDGE_REQUIRED"
    WEBSITE_KNOWLEDGE_REQUIRED = "WEBSITE_KNOWLEDGE_REQUIRED"
    STRUCTURED_QA_REQUIRED = "STRUCTURED_QA_REQUIRED"
    WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED = "WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED"
    COMBINED_KNOWLEDGE_REQUIRED = "COMBINED_KNOWLEDGE_REQUIRED"


class KnowledgeRouter:
    """
    Deterministic intent router for structured Q&A, website, and project knowledge.
    """

    # Exact chit-chat / greeting phrases that NEVER require database search
    CHIT_CHAT_EXACT = {
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "thanks a lot", "bye", "goodbye", "see you", "ok", "okay",
        "what is my name", "who are you", "what can you do", "help", "how are you"
    }

    # Short follow-up question triggers
    FOLLOW_UP_TRIGGERS = [
        "in which time", "what time", "timing", "timings", "how much", "fees", "cost",
        "where", "how long", "duration", "what about that", "and fees", "when", "schedule",
        "what documents", "which documents", "required documents", "documents required"
    ]

    # Project domain trigger keywords
    PROJECT_TRIGGERS = [
        "project", "projects", "status", "manager", "who manages", "who is managing",
        "client", "p001", "p002", "p003", "p004", "p005", "alpha", "beta", "gamma",
        "start date", "end date", "deadline", "lead", "owner"
    ]

    # Website domain trigger keywords — covers preciousedu.in service offerings
    WEBSITE_TRIGGERS = [
        # Services & offerings
        "service", "services", "coaching", "counselling", "counseling", "assistance",
        "guidance", "support", "consultation",
        # Study abroad topics
        "study abroad", "study in", "abroad", "overseas education", "foreign education",
        "university", "college", "admission", "course", "program", "degree",
        # Countries
        "usa", "uk", "canada", "australia", "germany", "new zealand", "ireland",
        "france", "singapore", "malaysia", "netherlands", "sweden",
        # Visa topics
        "visa", "student visa", "work visa", "f1", "f-1", "m1", "m-1", "tourist visa",
        "immigration", "pr", "permanent residency",
        # Test prep
        "ielts", "toefl", "pte", "gre", "gmat", "sat", "act", "test prep",
        "english test", "language test",
        # Fees & finance
        "fee", "fees", "tuition", "scholarship", "financial aid", "cost",
        # Company info
        "preciousedu", "precious education", "precious edu", "preciousedu.in", "peic", "piec",
        "office", "branch", "contact", "address", "location", "phone", "email",
        "about precious", "who is precious",
        # Eligibility & documents
        "eligibility", "requirement", "document", "documents", "sop", "letter of recommendation",
        "lor", "transcript", "application process", "apply",
        # Accommodation & logistics
        "accommodation", "hostel", "stay", "living", "housing",
    ]

    # Regex for Project ID matching (e.g., P001, P-002, PROJ-123, P1)
    PROJECT_ID_REGEX = re.compile(r"\b(P\d{1,4}|P-\d{1,4}|PROJ-\d{1,4}|P[0-9]{3})\b", re.IGNORECASE)

    def classify_intent(
        self,
        user_message: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None
    ) -> IntentClassificationResult:
        """
        Classifies intent deterministically and resolves follow-up queries using recent context.
        """
        raw_msg = user_message.strip()
        cleaned_msg = raw_msg.lower()

        # 1. Greeting & Identity detection
        if any(p in cleaned_msg for p in ["who are you", "what are you", "who is precious", "what is precious ai", "introduce yourself"]):
            return IntentClassificationResult("greetings", 0.98, raw_msg)
        if cleaned_msg in self.CHIT_CHAT_EXACT or any(cleaned_msg.startswith(g) for g in ["hello", "hi ", "hey ", "good morning", "good evening"]):
            return IntentClassificationResult("greeting", 1.00, raw_msg)

        # 2. Acronym & Full form detection
        if re.search(r"\b(spp|ssp)\b", cleaned_msg):
            return IntentClassificationResult("fullform_spp", 0.98, raw_msg)
        if re.search(r"\b(peic|piec)\b", cleaned_msg):
            return IntentClassificationResult("fullform_peic", 0.98, raw_msg)

        # 3. Contact information detection
        if any(w in cleaned_msg for w in ["email", "mail id", "email address", "contact email"]):
            return IntentClassificationResult("contact_email", 0.98, raw_msg)
        if any(w in cleaned_msg for w in ["mobile", "phone", "mobile number", "phone number", "contact number", "call us", "call you"]):
            return IntentClassificationResult("contact_phone", 0.98, raw_msg)
        if any(w in cleaned_msg for w in ["who is the owner", "who is owner", "founder", "director", "who owns", "owner"]):
            return IntentClassificationResult("about_us_general", 0.95, raw_msg)
        if any(w in cleaned_msg for w in ["contact info", "contact details", "contact us", "reach you", "office address", "head office", "location"]):
            if not any(w in cleaned_msg for w in ["visa", "ielts", "student", "study"]):
                return IntentClassificationResult("contact_general", 0.95, raw_msg)

        # 4. Multi-turn follow-up resolution (ONLY for genuine context-dependent follow-ups)
        is_followup = any(trig in cleaned_msg for trig in self.FOLLOW_UP_TRIGGERS) or cleaned_msg in [
            "tell me how", "how", "how to apply", "when", "where", "how much", "what time", "in which time", "and fees"
        ]
        if is_followup and recent_messages:
            prev_topic, prev_ctx_str = self._extract_previous_topic(recent_messages)
            if prev_topic == "ielts":
                if any(w in cleaned_msg for w in ["time", "timing", "timings", "when", "duration", "schedule", "long", "commitment"]):
                    resolved = "what is the total time commitment for ielts?"
                    return IntentClassificationResult("ielts_module_duration", 0.95, resolved, prev_ctx_str)
                elif any(w in cleaned_msg for w in ["how", "join", "enroll", "process", "register", "start", "fee", "fees"]):
                    resolved = "what is the process to join ielts coaching?"
                    return IntentClassificationResult("service_ielts", 0.95, resolved, prev_ctx_str)
                else:
                    resolved = f"IELTS coaching: {raw_msg}"
                    return IntentClassificationResult("service_ielts", 0.92, resolved, prev_ctx_str)

            elif prev_topic in ("student_visa", "australia", "canada", "uk", "usa"):
                if any(w in cleaned_msg for w in ["document", "documents", "need", "needed", "require", "required", "checklist"]):
                    resolved = "what documents are needed for university admission applications?"
                    return IntentClassificationResult("study_abroad_documents", 0.95, resolved, prev_ctx_str)
                elif any(w in cleaned_msg for w in ["how", "process", "apply"]):
                    country_intent = f"visa_{prev_topic}_student" if prev_topic in ("australia", "canada", "usa") else "countries_offered_uk" if prev_topic == "uk" else "service_student_visas"
                    return IntentClassificationResult(country_intent, 0.95, f"{prev_topic} student visa process", prev_ctx_str)

            elif prev_topic == "services":
                resolved = "what services do you provide?"
                return IntentClassificationResult("about_us_services", 0.92, resolved, prev_ctx_str)

        # 5. Entity Extraction: Country + Visa Type Matrix
        from app.knowledge.structured_qa_normalizer import StructuredQANormalizer
        detected_country = StructuredQANormalizer.detect_country(cleaned_msg)
        detected_visa = StructuredQANormalizer.detect_visa_type(cleaned_msg)

        if detected_country:
            country_key = detected_country.lower().replace(" ", "_")
            if detected_visa == "visitor":
                return IntentClassificationResult(f"visa_{country_key}_visitor", 0.98, raw_msg)
            elif detected_visa == "student":
                if detected_country == "UK":
                    return IntentClassificationResult("visa_uk_student", 0.98, raw_msg)
                return IntentClassificationResult(f"visa_{country_key}_student", 0.98, raw_msg)
            elif detected_visa == "work":
                return IntentClassificationResult(f"visa_{country_key}_work", 0.98, raw_msg)
            else:
                if detected_country == "UK":
                    return IntentClassificationResult("countries_offered_uk", 0.95, raw_msg)
                return IntentClassificationResult(f"visa_{country_key}_general", 0.95, raw_msg)

        # 6. IELTS / Test prep detection
        if any(re.search(r'\b' + re.escape(w) + r'\b', cleaned_msg) for w in ["ielts", "toefl", "pte", "gre", "gmat"]):
            if any(w in cleaned_msg for w in ["time", "timing", "duration", "how long", "schedule", "commitment"]):
                return IntentClassificationResult("ielts_module_duration", 0.95, raw_msg)
            if any(w in cleaned_msg for w in ["how", "join", "enroll", "process", "register", "start"]):
                return IntentClassificationResult("service_ielts", 0.95, raw_msg)
            return IntentClassificationResult("ielts_class", 0.95, raw_msg)

        # 7. Documents detection
        if any(re.search(r'\b' + re.escape(w) + r'\b', cleaned_msg) for w in ["document", "documents", "transcripts", "transcript", "sop", "lor"]):
            return IntentClassificationResult("study_abroad_documents", 0.95, raw_msg)

        # 8. Specific Visa Types without country
        if detected_visa == "visitor":
            return IntentClassificationResult("imm_visitor_visa_services", 0.95, raw_msg)
        if detected_visa == "student" or "student visa" in cleaned_msg or "study visa" in cleaned_msg or "study permit" in cleaned_msg:
            return IntentClassificationResult("service_student_visas", 0.95, raw_msg)
        if detected_visa == "work":
            return IntentClassificationResult("general_work_permit_services", 0.95, raw_msg)
        if detected_visa == "immigration":
            return IntentClassificationResult("imm_visa_services_overview", 0.95, raw_msg)

        # 9. Services detection
        service_keywords = ["service", "services", "offerings", "counseling", "counselling", "what services", "what do you provide"]
        if any(w in cleaned_msg for w in service_keywords):
            return IntentClassificationResult("about_us_services", 0.95, raw_msg)

        # 10. General Visa / Immigration detection
        visa_keywords = ["visa", "immigration", "pr", "permit", "abroad", "overseas", "admission", "admissions"]
        if any(re.search(r'\b' + re.escape(w) + r'\b', cleaned_msg) for w in visa_keywords):
            return IntentClassificationResult("visa_services", 0.90, raw_msg)

        # 11. Legal / Privacy policy / Terms detection
        if any(w in cleaned_msg for w in ["privacy", "privacy policy", "personal data"]):
            return IntentClassificationResult("privacy_policy", 0.95, raw_msg)
        if any(w in cleaned_msg for w in ["terms", "terms and conditions", "terms of service"]):
            return IntentClassificationResult("terms_conditions", 0.95, raw_msg)

        # 12. Fallback / Unknown
        return IntentClassificationResult("unknown", 0.20, raw_msg)

    def _extract_previous_topic(self, recent_messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Scans recent conversation history for previous turn topic."""
        for msg in reversed(recent_messages):
            content = msg.get("content", "").lower()
            if any(w in content for w in ["ielts", "coaching", "toefl", "pte"]):
                return "ielts", f"Previous turn topic: IELTS ('{msg.get('content', '')}')"
            if "australia" in content:
                return "australia", f"Previous turn topic: Australia ('{msg.get('content', '')}')"
            if "canada" in content:
                return "canada", f"Previous turn topic: Canada ('{msg.get('content', '')}')"
            if "uk" in content:
                return "uk", f"Previous turn topic: UK ('{msg.get('content', '')}')"
            if "usa" in content or "united states" in content or "america" in content:
                return "usa", f"Previous turn topic: USA ('{msg.get('content', '')}')"
            if any(w in content for w in ["student visa", "study visa", "study permit", "student permit"]):
                return "student_visa", f"Previous turn topic: Student Visa ('{msg.get('content', '')}')"
            if any(w in content for w in ["service", "services"]):
                return "services", f"Previous turn topic: Services ('{msg.get('content', '')}')"
        return None, None

    def route(
        self,
        user_message: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[RoutingDecision, Optional[StructuredProjectQuery]]:
        """
        Evaluate user message and history to determine routing.
        """
        cleaned_msg = user_message.strip().lower()

        # 1. Immediate chit-chat bypass
        if cleaned_msg in self.CHIT_CHAT_EXACT:
            return RoutingDecision.NO_KNOWLEDGE_REQUIRED, None

        # 2. Check for explicit project ID (e.g., P001)
        pid_match = self.PROJECT_ID_REGEX.search(user_message)
        if pid_match:
            pid = pid_match.group(1).upper()
            query = StructuredProjectQuery(
                entity="project",
                operation="get",
                filters={"project_id": pid},
                fields=[]
            )
            return RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED, query

        # 3. Evaluate trigger sets using word boundaries
        has_project_trigger = any(re.search(r'\b' + re.escape(trig) + r'\b', cleaned_msg) for trig in self.PROJECT_TRIGGERS)
        has_website_trigger = any(re.search(r'\b' + re.escape(trig) + r'\b', cleaned_msg) for trig in self.WEBSITE_TRIGGERS)

        # Check for multi-turn follow-up reference
        intent_res = self.classify_intent(user_message, recent_messages)
        if intent_res.predicted_intent in ["ielts_classes", "services"]:
            has_website_trigger = True

        resolved_project = None
        has_referential_pronoun = bool(re.search(r"\b(it|its|this|that)\b", cleaned_msg)) and any(
            q in cleaned_msg for q in ["who", "what", "status", "manager", "client", "manage", "lead"]
        )
        if has_referential_pronoun and recent_messages:
            resolved_project = self._extract_recent_project_entity(recent_messages)

        # 4. Combined routing decision
        if (has_project_trigger or resolved_project) and has_website_trigger:
            query = self._build_project_query(user_message, cleaned_msg, resolved_project)
            return RoutingDecision.WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED, query

        if has_website_trigger and not has_project_trigger and not resolved_project:
            return RoutingDecision.WEBSITE_KNOWLEDGE_REQUIRED, None

        if not has_project_trigger and not resolved_project:
            return RoutingDecision.NO_KNOWLEDGE_REQUIRED, None

        query = self._build_project_query(user_message, cleaned_msg, resolved_project)
        return RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED, query

    def _build_project_query(
        self,
        user_message: str,
        cleaned_msg: str,
        resolved_project: Optional[str]
    ) -> StructuredProjectQuery:
        """Builds a StructuredProjectQuery from message context."""
        filters: Dict[str, Any] = {}

        if resolved_project:
            if self.PROJECT_ID_REGEX.match(resolved_project):
                filters["project_id"] = resolved_project
            else:
                filters["project_name"] = resolved_project
        else:
            extracted_name = self._extract_project_name(user_message)
            if extracted_name:
                filters["project_name"] = extracted_name
            else:
                filters["query"] = user_message.strip()

        fields = []
        if "status" in cleaned_msg:
            fields.append("status")
        if "manager" in cleaned_msg or "manages" in cleaned_msg or "managing" in cleaned_msg:
            fields.append("manager")
        if "client" in cleaned_msg:
            fields.append("client")
        if "date" in cleaned_msg or "start" in cleaned_msg or "end" in cleaned_msg:
            fields.extend(["start_date", "end_date"])

        return StructuredProjectQuery(
            entity="project",
            operation="get",
            filters=filters,
            fields=fields
        )

    def _extract_recent_project_entity(self, recent_messages: List[Dict[str, Any]]) -> Optional[str]:
        """Search recent conversation history for previously mentioned project ID or name."""
        for msg in reversed(recent_messages):
            content = msg.get("content", "")
            pid_match = self.PROJECT_ID_REGEX.search(content)
            if pid_match:
                return pid_match.group(1).upper()

            name_match = re.search(r"Project\s+([A-Za-z0-9_-]+)", content, re.IGNORECASE)
            if name_match:
                return name_match.group(1)

        return None

    def _extract_project_name(self, text: str) -> Optional[str]:
        """Extract explicit project name from phrases like 'Project Alpha' or 'project Alpha'."""
        match = re.search(r"Project\s+([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
