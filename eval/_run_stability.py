import json, os
from pathlib import Path

# Set temperature to 0.0 BEFORE any model-building imports
from app.config import get_settings
settings = get_settings()
settings.OPENAI_MODEL_TEMPERATURE = 0.0

from eval.regression_suite import load_golden_set, lcel_router
from app.rag.qa_chain import reset_llm_cache
from eval.custom_metrics import compute_answer_stability, SemanticSimilarityScorer

N = int(os.getenv("STABILITY_SAMPLE_SIZE", "10"))
reset_llm_cache()
scorer = SemanticSimilarityScorer()

a, b = [], []
for c in load_golden_set()["cases"][:N]:
    q = c.get("query","")
    try:
        ra = lcel_router.invoke({"session_id": f"stab-a-{c.get('id','x')}", "user_message": q})
        rb = lcel_router.invoke({"session_id": f"stab-b-{c.get('id','x')}", "user_message": q})
        a.append(ra.get("answer_text","")); b.append(rb.get("answer_text",""))
    except Exception as e:
        print("skip", c.get("id"), type(e).__name__, e); continue

res = compute_answer_stability(a, b, scorer=scorer)
Path("reports").mkdir(exist_ok=True)
Path("reports/_stability.json").write_text(json.dumps(res, indent=2))
print("stability =", res["stability_score"])