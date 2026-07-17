"""MCP stub server: IRDAI Compliance API (port 9004)."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict
import base64

app = FastAPI(title="IRDAI Compliance API MCP Server")

# Basic auth credentials
EXPECTED_USERNAME = "irdai-service"
EXPECTED_PASSWORD = "test-irdai-pass-2024"


def _verify_auth(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode()
        username, password = decoded.split(":", 1)
        return username == EXPECTED_USERNAME and password == EXPECTED_PASSWORD
    except Exception:
        return False


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


# Compliance status data
COMPLIANCE_DATA = {
    "claim": {
        "C1001": {
            "compliance_status": "compliant",
            "last_review_date": "2024-12-01",
            "reviewer": "IRDAI-AUTO-001",
            "notes": "Standard claim, no regulatory issues",
            "reporting_required": False,
        },
        "C2001": {
            "compliance_status": "needs_review",
            "last_review_date": "2024-12-15",
            "reviewer": "IRDAI-AUTO-002",
            "notes": "High-value claim requires additional documentation per IRDAI circular 2024/12",
            "reporting_required": True,
            "reporting_deadline": "2025-01-30",
        },
        "C2002": {
            "compliance_status": "non_compliant",
            "last_review_date": "2024-12-10",
            "reviewer": "IRDAI-AUTO-003",
            "notes": "Missing required hospitalization documents. Non-compliance notice issued.",
            "reporting_required": True,
            "reporting_deadline": "2025-01-15",
            "penalty_risk": "medium",
        },
    },
    "policy": {
        "P123456": {
            "compliance_status": "compliant",
            "last_review_date": "2024-06-01",
            "reviewer": "IRDAI-AUTO-001",
            "notes": "Policy meets all regulatory requirements",
            "reporting_required": False,
        },
        "P789012": {
            "compliance_status": "compliant",
            "last_review_date": "2024-07-15",
            "reviewer": "IRDAI-AUTO-001",
            "notes": "High-value policy, quarterly reporting required",
            "reporting_required": True,
            "reporting_deadline": "2025-01-15",
        },
        "P654321": {
            "compliance_status": "compliant",
            "last_review_date": "2024-03-01",
            "reviewer": "IRDAI-AUTO-002",
            "notes": "Lapsed policy, no active compliance issues",
            "reporting_required": False,
        },
    },
}

REPORTING_REQUIREMENTS = {
    "hospitalization": {
        "forms": ["IRDAI-HF-001", "IRDAI-HF-002"],
        "documentation_required": ["discharge_summary", "itemized_bill", "admission_notes"],
        "submission_window_days": 30,
        "regulatory_body": "IRDAI Health Insurance Division",
    },
    "surgery": {
        "forms": ["IRDAI-SF-001", "IRDAI-SF-002", "IRDAI-SF-003"],
        "documentation_required": ["pre_auth_form", "surgery_notes", "discharge_summary", "pathology_report"],
        "submission_window_days": 15,
        "regulatory_body": "IRDAI Health Insurance Division",
    },
    "outpatient": {
        "forms": ["IRDAI-OP-001"],
        "documentation_required": ["prescription", "consultation_notes", "pharmacy_receipt"],
        "submission_window_days": 45,
        "regulatory_body": "IRDAI Health Insurance Division",
    },
    "maternity": {
        "forms": ["IRDAI-MF-001", "IRDAI-MF-002"],
        "documentation_required": ["admission_notes", "delivery_report", "discharge_summary", "newborn_details"],
        "submission_window_days": 60,
        "regulatory_body": "IRDAI Maternity Benefits Division",
    },
}


@app.get("/health")
def health(request: Request):
    if not _verify_auth(request):
        return _unauthorized()
    return {"status": "ok", "server": "irdai_compliance"}


@app.get("/tools")
def list_tools(request: Request):
    if not _verify_auth(request):
        return _unauthorized()
    return {
        "tools": [
            {
                "name": "check_compliance_status",
                "description": "Check IRDAI compliance status for a claim or policy.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "enum": ["claim", "policy"],
                            "description": "Type of entity to check",
                        },
                        "entity_id": {
                            "type": "string",
                            "description": "Claim ID or Policy number",
                        },
                    },
                    "required": ["entity_type", "entity_id"],
                },
            },
            {
                "name": "get_reporting_requirements",
                "description": "Get IRDAI regulatory reporting requirements for a specific claim type.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "claim_type": {
                            "type": "string",
                            "description": "Type of claim",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Claim amount",
                        },
                    },
                    "required": ["claim_type"],
                },
            },
        ]
    }


class InvokeRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}


@app.post("/invoke")
def invoke(req: InvokeRequest, request: Request):
    if not _verify_auth(request):
        return _unauthorized()

    if req.tool == "check_compliance_status":
        entity_type = req.arguments.get("entity_type", "")
        entity_id = req.arguments.get("entity_id", "")

        if entity_type not in ("claim", "policy"):
            return {"error": f"Invalid entity_type: '{entity_type}'. Must be 'claim' or 'policy'"}

        entity_data = COMPLIANCE_DATA.get(entity_type, {})
        result = entity_data.get(entity_id)
        if not result:
            return {
                "error": f"{entity_type.capitalize()} '{entity_id}' not found",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "found": False,
            }

        return {"entity_type": entity_type, "entity_id": entity_id, **result, "found": True}

    elif req.tool == "get_reporting_requirements":
        claim_type = req.arguments.get("claim_type", "").lower()
        amount = req.arguments.get("amount", 0.0)

        requirements = REPORTING_REQUIREMENTS.get(claim_type)
        if not requirements:
            return {
                "error": f"No reporting requirements found for claim type '{claim_type}'",
                "claim_type": claim_type,
                "found": False,
            }

        # Adjust submission window for high-value claims
        result = dict(requirements)
        if amount > 100000:
            result["submission_window_days"] = max(7, result["submission_window_days"] - 5)
            result["expedited"] = True
            result["notes"] = "High-value claim: expedited reporting required"
        else:
            result["expedited"] = False

        result["claim_type"] = claim_type
        result["amount"] = amount
        result["found"] = True
        return result

    return {"error": f"Unknown tool: {req.tool}"}


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=9004, log_level="info")


if __name__ == "__main__":
    main()