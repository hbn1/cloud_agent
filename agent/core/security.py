"""Security guardrails — input sanitization and PII detection.

L1 Defense: regex-based prompt injection detection + PII redaction.
L2 Defense: MCP tool RBAC (implemented in mcp_pool.py).
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# L1: Prompt Injection patterns (regex-based, catches script-kiddie attacks)
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    # "Ignore all previous instructions"
    r"(?i)\b(ignore|forget|disregard)\s+(all|everything|every)\s+(previous|prior|above)\s+(instructions?|prompts?|rules?)\b",
    # "You are now..."
    r"(?i)\byou\s+are\s+now\s+(DAN|jailbreak|a\s+different\s+AI|evil)",
    # System prompt extraction attempts
    r"(?i)\b(print|show|reveal|display|output|echo)\s+(your|the|the\s+system)\s+(prompt|instructions?|system\s+message)\b",
    # "Act as..." role hijacking
    r"(?i)\b(from\s+now\s+on\s+)?act\s+as\s+(a\s+)?(different|another|new)\b",
    # Token/Session stuffing
    r"(?i)\b\[SYSTEM\]|\[/SYSTEM\]|<<SYS>>|<\|im_start\|>|<\|im_end\|>",
    # Unicode confusables / homoglyph attacks (basic)
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e]",  # zero-width & bidi control chars
]

# ---------------------------------------------------------------------------
# L1: PII patterns for detection/redaction
# ---------------------------------------------------------------------------
PII_PATTERNS = {
    "phone": r"\b1[3-9]\d{9}\b",                    # Chinese mobile
    "email": r"\b[\w.-]+@[\w.-]+\.\w{2,}\b",
    "id_card": r"\b\d{6}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "ip_addr": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


class SecurityGate:
    """Apply L1 security checks before the query enters LangGraph."""

    @staticmethod
    def sanitize(query: str) -> tuple[str, list[str], bool]:
        """Sanitize user input. Returns (clean_query, flags, blocked).

        Args:
            query: Raw user input string.

        Returns:
            (sanitized_query, list_of_triggered_rules, is_blocked)
        """
        flags: list[str] = []
        blocked = False

        # Check for prompt injection
        for i, pattern in enumerate(INJECTION_PATTERNS):
            if re.search(pattern, query):
                flags.append(f"injection_pattern_{i}")
                blocked = True
                logger.warning("SecurityGate: blocked injection pattern #%d in query", i)

        # Check for PII
        pii_found = ""
        for pii_type, pattern in PII_PATTERNS.items():
            match = re.search(pattern, query)
            if match:
                pii_found = match.group()
                flags.append(f"pii_{pii_type}")
                # Redact PII
                query = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", query)
                logger.info("SecurityGate: redacted %s from query", pii_type)

        return query, flags, blocked

    @staticmethod
    def redact_pii(text: str) -> str:
        """Redact all PII from text (for logging)."""
        for pii_type, pattern in PII_PATTERNS.items():
            text = re.sub(pattern, f"[{pii_type.upper()}]", text)
        return text


security_gate = SecurityGate()