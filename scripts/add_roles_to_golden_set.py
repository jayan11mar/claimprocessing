#!/usr/bin/env python3
"""
Add realistic role assignments to all 200 golden set cases.
Role distribution target: ~60% customer, remainder spread across other roles.
"""
import json
from pathlib import Path

def assign_roles():
    # Load the golden set
    golden_set_path = Path(__file__).parent.parent / "eval" / "golden_set.json"
    with open(golden_set_path, 'r') as f:
        data = json.load(f)
    
    # Define role assignments per project
    # Based on query analysis:
    # - customer svc: All customer-facing (billing, refunds, follow-ups)
    # - claims / insurance: Mostly customer, 8 regulatory as compliance_officer
    # - loan underwriting: All underwriter (risk assessment, eligibility, pricing)
    # - aml / fraud: Mix of customer, claims_adjuster, and compliance_officer
    
    role_configs = {
        "customer svc": {
            "role": "customer",
            "count": 50
        },
        "claims / insurance": {
            "role": "customer",
            "count": 42,
            "regulatory_indices": [3, 8, 13, 18, 23, 28, 33, 38],  # 0-indexed: RAG-03, 08, 13, 18, 23, 28, 33, 38
            "regulatory_role": "compliance_officer"
        },
        "loan underwriting": {
            "role": "underwriter",
            "count": 50
        },
        "aml / fraud": {
            "customer_count": 28,
            "claims_adjuster_count": 12,
            "compliance_officer_count": 10
        }
    }
    
    total_modified = 0
    
    for project in data["projects"]:
        project_name = project["name"]
        items = project["items"]
        
        if project_name == "customer svc":
            # All customer
            for item in items:
                if "role" not in item:
                    item["role"] = "customer"
                    total_modified += 1
                    
        elif project_name == "claims / insurance":
            # Mostly customer, 8 regulatory as compliance_officer
            config = role_configs[project_name]
            for idx, item in enumerate(items):
                if "role" not in item:
                    if idx in config["regulatory_indices"]:
                        item["role"] = config["regulatory_role"]
                    else:
                        item["role"] = config["role"]
                    total_modified += 1
                    
        elif project_name == "loan underwriting":
            # All underwriter
            for item in items:
                if "role" not in item:
                    item["role"] = "underwriter"
                    total_modified += 1
                    
        elif project_name == "aml / fraud":
            # Mix: 28 customer, 12 claims_adjuster, 10 compliance_officer
            config = role_configs[project_name]
            customer_end = config["customer_count"]
            claims_adjuster_end = customer_end + config["claims_adjuster_count"]
            
            for idx, item in enumerate(items):
                if "role" not in item:
                    if idx < customer_end:
                        item["role"] = "customer"
                    elif idx < claims_adjuster_end:
                        item["role"] = "claims_adjuster"
                    else:
                        item["role"] = "compliance_officer"
                    total_modified += 1
    
    # Write back the updated JSON
    with open(golden_set_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Modified {total_modified} items")
    
    # Verify the changes
    verify_roles()

def verify_roles():
    """Verify that all cases have roles and distribution is correct."""
    golden_set_path = Path(__file__).parent.parent / "eval" / "golden_set.json"
    with open(golden_set_path, 'r') as f:
        data = json.load(f)
    
    from collections import Counter
    
    all_items = []
    for project in data["projects"]:
        all_items.extend(project["items"])
    
    missing = [c.get('id') for c in all_items if not c.get('role')]
    role_dist = Counter(c.get('role') for c in all_items)
    
    print(f"\nTotal cases: {len(all_items)}")
    print(f"Missing role: {len(missing)}")
    print(f"Role distribution: {dict(role_dist)}")
    print(f"Role distribution %: { {k: f'{v/len(all_items)*100:.1f}%' for k, v in role_dist.items()} }")
    
    if missing:
        print(f"\n❌ FAILED: {len(missing)} cases still missing roles!")
        return False
    
    if len(role_dist) < 3:
        print(f"\n❌ FAILED: Only {len(role_dist)} role types found, need at least 3!")
        return False
    
    print("\n✅ SUCCESS: All cases have roles with meaningful distribution!")
    return True

if __name__ == "__main__":
    assign_roles()