from app.tools.policy_checker import check_policy_status, PolicyCheckResult


def test_check_policy_status_active():
    result = check_policy_status("P123456")
    assert isinstance(result, PolicyCheckResult)
    assert result.policy_number == "P123456"
    assert result.status == "ACTIVE"
    assert result.is_active is True
    assert "ACTIVE" in result.message
    assert "can file claims" in result.message


def test_check_policy_status_lapsed():
    result = check_policy_status("P654321")
    assert isinstance(result, PolicyCheckResult)
    assert result.policy_number == "P654321"
    assert result.status == "LAPSED"
    assert result.is_active is False
    assert "LAPSED" in result.message
    assert "cannot be filed" in result.message


def test_check_policy_status_not_found():
    result = check_policy_status("P999999")
    assert isinstance(result, PolicyCheckResult)
    assert result.policy_number == "P999999"
    assert result.status == "NOT_FOUND"
    assert result.is_active is False
    assert "not found" in result.message.lower()
    assert result.details.get("error") == "policy_not_found"


def test_check_policy_status_empty_policy_number():
    result = check_policy_status("")
    assert isinstance(result, PolicyCheckResult)
    assert result.status == "UNKNOWN"
    assert result.is_active is False
    assert "required" in result.message.lower()
    assert result.details.get("error") == "policy_number_missing"


def test_check_policy_status_whitespace_only():
    result = check_policy_status("   ")
    assert isinstance(result, PolicyCheckResult)
    assert result.status == "UNKNOWN"
    assert result.is_active is False
    assert "required" in result.message.lower()


def test_check_policy_status_includes_details():
    result = check_policy_status("P123456")
    assert result.details is not None
    assert "status" in result.details
    assert "sum_insured" in result.details
    assert "deductible" in result.details
    assert "start_date" in result.details
    assert "end_date" in result.details
    assert result.details["status"] == "ACTIVE"


def test_check_policy_status_result_to_dict():
    result = check_policy_status("P123456")
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict["policy_number"] == "P123456"
    assert result_dict["is_active"] is True
    assert "message" in result_dict
    assert "details" in result_dict