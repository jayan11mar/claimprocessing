import json
import time
from uuid import uuid4

import requests


API_URL = "http://localhost:8000"

QUERIES = [
    "Register a new health insurance claim for policy #HI-550012.",
    "What documents are needed for a motor accident claim?",
    "Check if surgery X is covered under policy #HI-445021.",
    "What is the fraud score for claim #CLM-90210?",
    "Calculate settlement for a claim of ₹5,60,000 with 10K deductible.",
    "Why was my claim partially rejected?",
    "Show me the claim history for policyholder ID P-3321.",
    "Is pre-hospitalization covered for this policy?",
    "Flag duplicate claims across family floater policies.",
    "What is the average processing time for this claim type?",
    "Escalate claim #CLM-77654 — it’s been pending 20 days.",
    "Compare claimed amount vs. policy sub-limits.",
    "Generate a settlement breakdown for claim #CLM-88712.",
    "What exclusions apply to this policy?",
    "Check if the hospital is in the network list.",
    "How many claims has this policyholder filed in 2 years?",
    "Validate the diagnosis code against the treatment billed.",
    "What is the co-pay percentage for this plan?",
    "Draft a claim rejection letter with reasons.",
    "Summarize all pending claims in my review queue.",
]


def run_evaluation():
    session_id = uuid4().hex
    results = []
    for q in QUERIES:
        payload = {"session_id": session_id, "message": q}
        try:
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            data = {"error": str(exc)}
        except ValueError as exc:
            data = {"error": f"Invalid JSON response: {exc}"}

        entry = {"query": q, "response": data}
        results.append(entry)
        print(f"Sent: {q[:60]}... -> received: {('error' in data) and 'ERROR' or 'OK'}")
        time.sleep(0.25)

    with open("scripts/results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print("Saved results to scripts/results.json")


if __name__ == "__main__":
    run_evaluation()
