"""MCP stub server: Policy Administration System (port 9002)."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(title="Policy Administration System MCP Server")

# Expected API key
EXPECTED_API_KEY = "test-policy-api-key-2024"

# In-memory policy data
POLICIES = {
    "P123456": {
        "policy_number": "P123456",
        "policy_holder_id": "H1001",
        "status": "ACTIVE",
        "sum_insured": 10000.0,
        "deductible": 500.0,
        "copay_percent": 10.0,
        "start_date": "2024-01-01",
        "end_date": "2027-01-01",
        "sub_limits": {"hospital": 5000.0, "dental": 1000.0},
        "depreciation_schedule": {"hospital": 10.0, "dental": 20.0},
    },
    "P789012": {
        "policy_number": "P789012",
        "policy_holder_id": "H1001",
        "status": "ACTIVE",
        "sum_insured": 500000.0,
        "deductible": 10000.0,
        "copay_percent": 20.0,
        "start_date": "2024-02-01",
        "end_date": "2028-02-01",
        "sub_limits": {"hospital": 100000.0, "surgery": 150000.0},
        "depreciation_schedule": {"hospital": 5.0, "surgery": 15.0},
    },
    "P654321": {
        "policy_number": "P654321",
        "policy_holder_id": "H1002",
        "status": "LAPSED",
        "sum_insured": 5000.0,
        "deductible": 250.0,
        "copay_percent": 20.0,
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "sub_limits": {"vision": 300.0},
        "depreciation_schedule": {"vision": 10.0},
    },
}

# Eligibility rules (simplified)
ELIGIBLE_DIAGNOSES = {
    "S75.1": {"description": "Injury of unspecified nerve", "coverage": 0.9},
    "T20.2": {"description": "Burn of second degree", "coverage": 0.85},
    "I10": {"description": "Essential hypertension", "coverage": 1.0},
    "J45": {"description": "Asthma", "coverage": 0.95},
    "M54.5": {"description": "Low back pain", "coverage": 0.8},
}


def _verify_auth(request: Request) -> bool:
    api_key = request.headers.get("X-API-Key", "")
    return api_key == EXPECTED_API_KEY


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


@app.get("/health")
def health(request: Request):
    if not _verify_auth(request):
        return _unauthorized()
    return {"status": "ok", "server": "policy_administration"}


@app.get("/tools")
def list_tools(request: Request):
    if not _verify_auth(request):
        return _unauthorized()
    return {
        "tools": [
            {
                "name": "get_policy_details",
                "description": "Get detailed policy information including coverage limits, deductibles, and copay.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "policy_number": {"type": "string", "description": "Policy number to look up"},
                    },
                    "required": ["policy_number"],
                },
            },
            {
                "name": "check_claim_eligibility",
                "description": "Check if a claim is eligible under the policy based on diagnosis code and treatment type.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "policy_number": {"type": "string", "description": "Policy number"},
                        "diagnosis_code": {"type": "string", "description": "Medical diagnosis code (ICD-10)"},
                        "treatment_type": {"type": "string", "description": "Type of treatment"},
                        "claim_amount": {"type": "number", "description": "Proposed claim amount"},
                    },
                    "required": ["policy_number", "diagnosis_code"],
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

    if req.tool == "get_policy_details":
        policy_number = req.arguments.get("policy_number", "")
        policy = POLICIES.get(policy_number)
        if not policy:
            return {"error": f"Policy '{policy_number}' not found", "found": False}
        return {"policy": policy, "found": True}

    elif req.tool == "check_claim_eligibility":
        policy_number = req.arguments.get("policy_number", "")
        diagnosis_code = req.arguments.get("diagnosis_code", "")
        treatment_type = req.arguments.get("treatment_type", "")
        claim_amount = req.arguments.get("claim_amount", 0.0)

        policy = POLICIES.get(policy_number)
        if not policy:
            return {"error": f"Policy '{policy_number}' not found", "eligible": False}

        if policy["status"] != "ACTIVE":
            return {
                "eligible": False,
                "policy_number": policy_number,
                "diagnosis_code": diagnosis_code,
                "reason": f"Policy status is '{policy['status']}'",
                "approved_amount": 0.0,
            }

        diagnosis_info = ELIGIBLE_DIAGNOSES.get(diagnosis_code)
        if not diagnosis_info:
            return {
                "eligible": True,
                "policy_number": policy_number,
                "diagnosis_code": diagnosis_code,
                "reason": "Diagnosis code not in predefined list, manual review required",
                "approved_amount": 0.0,
                "requires_manual_review": True,
            }

        coverage = diagnosis_info["coverage"]
        max_payable = min(claim_amount * coverage, policy["sum_insured"])
        approved = max(0.0, max_payable - policy["deductible"])
        copay = approved * (policy["copay_percent"] / 100.0)
        net_approved = approved - copay

        return {
            "eligible": True,
            "policy_number": policy_number,
            "diagnosis_code": diagnosis_code,
            "diagnosis_description": diagnosis_info["description"],
            "treatment_type": treatment_type or "not_specified",
            "claim_amount": claim_amount,
            "coverage_rate": coverage,
            "deductible": policy["deductible"],
            "copay_amount": copay,
            "approved_amount": round(net_approved, 2),
            "max_payable": round(max_payable, 2),
            "sub_limits": policy["sub_limits"],
        }

    return {"error": f"Unknown tool: {req.tool}"}


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=9002, log_level="info")


if __name__ == "__main__":
    main()