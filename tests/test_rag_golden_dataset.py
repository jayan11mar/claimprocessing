from pathlib import Path

from scripts.validate_golden_dataset import load_dataset


def test_rag_golden_dataset_files_exist_with_50_items_each():
    root = Path(__file__).resolve().parent.parent
    dataset_paths = [
        root / "data" / "golden_dataset" / "rag_loan_underwriting.json",
        root / "data" / "golden_dataset" / "rag_customer_svc.json",
        root / "data" / "golden_dataset" / "rag_aml_fraud.json",
        root / "data" / "golden_dataset" / "rag_claims_insurance.json",
    ]

    for dataset_path in dataset_paths:
        assert dataset_path.exists(), f"Missing golden dataset {dataset_path.name}"
        payload = load_dataset(dataset_path)
        assert payload["project"]
        assert payload["threshold_metrics"]
        assert len(payload["items"]) == 50

        for item in payload["items"]:
            assert item["query"].strip()
            assert item["expected_answer"].strip()
            assert item["expected_chunks"]
            assert item["difficulty"] in {"easy", "medium", "hard"}
