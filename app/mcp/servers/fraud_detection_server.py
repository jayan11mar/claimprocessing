"""MCP stub server: Fraud Detection Scoring (port 9003)."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI(title="Fraud Detection Scoring MCP Server")

EXPECTED_TOKEN = "test-fraud-token-2024"

# In-memory fraud scores
FRAUD_SCORES: Dict[str, Dict[str, Any]] = {}


def _verify_auth(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {EXPECTED_TOKEN}"


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


@app.get("/health")
def health(request: Request):
    if not _verify_auth(request):
        return _unauthorized()
    return {"status": "ok", "server": "fraud_detection"}


@app.get("/tools")
def list_tools(request: Request):
    if not _verify_auth(request):
        return _unauthorized()
    return {
        "tools": [
            {
                "name": "score_fraud_risk",
                "description": "Score a claim for fraud risk. Returns a score from 0 (low risk) to 1 (high risk) with signal breakdown.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "description": "Claim ID to evaluate"},
                        "policy_number": {"type": "string", "description": "Policy number"},
                        "claim_amount": {"type": "number", "description": "Claim amount"},
                        "diagnosis_code": {"type": "string", "description": "Diagnosis code"},
                    },
                    "required": ["claim_id", "claim_amount"],
                },
            },
            {
                "name": "get_fraud_signals",
                "description": "Get detailed fraud signal analysis for a previously scored claim.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "description": "Claim ID to get fraud signals for"},
                    },
                    "required": ["claim_id"],
                },
            },
        ]
    }


class InvokeRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}


def _compute_fraud_score(claim_id: str, claim_amount: float, diagnosis_code: str, policy_number: str) -> Dict[str, Any]:
    """Compute a deterministic fraud score based on claim characteristics."""
    signals: list[str] = []
    score = 0.0

    # Signal 1: High claim amount
    if claim_amount > 100000:
        score += 0.3
        signals.append("high_claim_amount")
    elif claim_amount > 50000:
        score += 0.15
        signals.append("elevated_claim_amount")

    # Signal 2: Specific diagnosis codes that may indicate higher fraud risk
    high_risk_codes = {"M54.5", "S75.1"}  # back pain, nerve injury
    if diagnosis_code in high_risk_codes:
        score += 0.2
        signals.append("high_risk_diagnosis")

    # Signal 3: Round amounts may indicate fraud
    if claim_amount > 0 and claim_amount % 1000 == 0:
        score += 0.1
        signals.append("round_amount_suspicious")

    # Signal 4: Missing diagnosis code
    if not diagnosis_code or diagnosis_code.strip() == "":
        score += 0.25
        signals.append("missing_diagnosis_code")

    # Signal 5: Policy history (simulated)
    if policy_number == "P654321":
        score += 0.15
        signals.append("lapsed_policy")
    elif policy_number == "P123456":
        score += 0.05
        signals.append("moderate_risk_policy_profile")

    # Clamp to [0, 1]
    score = min(1.0, max(0.0, score))

    details = {
        "claim_amount_band": "high" if claim_amount > 100000 else ("medium" if claim_amount > 50000 else "low"),
        "diagnosis_risk": "high" if diagnosis_code in high_risk_codes else "low",
        "policy_status_risk": "elevated" if policy_number == "P654321" else "normal",
    }

    return {
        "claim_id": claim_id,
        "score": round(score, 4),
        "signals": signals,
        "details": details,
        "risk_level": "high" if score > 0.5 else ("medium" if score > 0.2 else "low"),
    }


@app.post("/invoke")
def invoke(req: InvokeRequest, request: Request):
    if not _verify_auth(request):
        return _unauthorized()

    if req.tool == "score_fraud_risk":
        claim_id = req.arguments.get("claim_id", "")
        claim_amount = float(req.arguments.get("claim_amount", 0))
        diagnosis_code = req.arguments.get("diagnosis_code", "")
        policy_number = req.arguments.get("policy_number", "")

        result = _compute_fraud_score(claim_id, claim_amount, diagnosis_code, policy_number)
        FRAUD_SCORES[claim_id] = result
        return result

    elif req.tool == "get_fraud_signals":
        claim_id = req.arguments.get("claim_id", "")
        result = FRAUD_SCORES.get(claim_id)
        if not result:
            return {"error": f"Claim '{claim_id}' not scored yet", "claim_id": claim_id, "scored": False}
        return {**result, "scored": True}

    return {"error": f"Unknown tool: {req.tool}"}


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=9003, log_level="info")


if __name__ == "__main__":
    main()