from typing import List, Dict


def get_faq_examples() -> List[Dict[str, str]]:
    return [
        {
            "user": "Process a new claim for policy #HI-445021. Hospital bill is ₹3,40,000.",
            "assistant": "Claim registered as #CLM-88712. Policy validated: Sum insured ₹5L, deductible ₹5,000, room rent sub-limit ₹4,000/day. After applying deductible and sub-limits, eligible settlement: ₹3,12,000. No fraud signals detected.",
            "intent": "CLAIM_REGISTRATION",
            "category": "claims",
        },
        {
            "user": "This claim looks suspicious — can you check?",
            "assistant": "Fraud analysis for #CLM-88010: 3 red flags detected — (1) claim filed within 45 days of policy purchase, (2) hospital not in network but in high-fraud zone, (3) similar claim amount filed by a related policy last quarter. Fraud score: 72/100. Recommending manual investigation.",
            "intent": "FRAUD_CHECK",
            "category": "fraud",
        },
        {
            "user": "What’s the status of my father’s claim?",
            "assistant": "Claim #CLM-71123 for policy #HI-330098 is currently in ‘Document Verification’ stage. We’re awaiting the original discharge summary. Once received, estimated processing time is 3–5 business days.",
            "intent": "CLAIM_STATUS",
            "category": "claims",
        },
        {
            "user": "What is the status of claim CLM-90210?",
            "assistant": "Claim CLM-90210 is currently in 'Document Verification' stage. We are awaiting the original discharge summary. Once received, estimated processing time is 3-5 business days.",
            "intent": "CLAIM_STATUS",
            "category": "claims",
        },
        {
            "user": "What is the fraud score for claim #CLM-90210?",
            "assistant": "The fraud score indicates risk based on claim history and anomalies. I can check the claim if the ID is valid and the fraud monitoring system has a recorded score.",
            "intent": "FRAUD_CHECK",
            "category": "fraud",
        },
        {
            "user": "Calculate settlement for a claim of ₹5,60,000 with 10K deductible.",
            "assistant": "With a ₹10,000 deductible, the claim amount subject to coverage is ₹5,50,000. The final settlement will then depend on co-pay, sub-limits, and approved claim items.",
            "intent": "SETTLEMENT_QUERY",
            "category": "settlement",
        },
        {
            "user": "Why was my claim partially rejected?",
            "assistant": "Partial rejection usually means part of the claim was outside coverage, above a sub-limit, or missing required documentation. I can review the specific reason if you share the claim details and rejection note.",
            "intent": "OTHER",
            "category": "claims",
        },
        {
            "user": "Show me the claim history for policyholder ID P-3321.",
            "assistant": "Claim history summaries include filed claims, statuses, settled amounts, and pending actions. Provide your policyholder ID and I can look up the relevant history records.",
            "intent": "POLICY_STATUS",
            "category": "history",
        },
        {
            "user": "Is pre-hospitalization covered for this policy?",
            "assistant": "Pre-hospitalization coverage is usually included for a defined number of days before admission, subject to policy benefits and documentation. Check your policy schedule or ask your insurer for the exact limit.",
            "intent": "OTHER",
            "category": "coverage",
        },
        {
            "user": "Flag duplicate claims across family floater policies.",
            "assistant": "Duplicate claim detection reviews whether the same incident has been claimed under multiple family floater policies. If records overlap, it is escalated for fraud review and verification.",
            "intent": "FRAUD_CHECK",
            "category": "fraud",
        },
        {
            "user": "What is the average processing time for this claim type?",
            "assistant": "Average processing time depends on the claim type and whether all required documents are submitted. Many standard health claims are reviewed within 5-10 business days after complete submission.",
            "intent": "OTHER",
            "category": "timing",
        },
        {
            "user": "Compare claimed amount vs. policy sub-limits.",
            "assistant": "When a claimed amount exceeds a policy sub-limit, coverage is typically capped at the sub-limit and the insured must pay the remainder. Sub-limits depend on the benefit category and policy terms.",
            "intent": "OTHER",
            "category": "policy",
        },
        {
            "user": "Draft a claim rejection letter with reasons.",
            "assistant": "A rejection letter should clearly state the claim number, the policy provisions relied on, the specific reasons for rejection, and any next steps for appeal or additional documentation.",
            "intent": "OTHER",
            "category": "claims",
        },
    ]
