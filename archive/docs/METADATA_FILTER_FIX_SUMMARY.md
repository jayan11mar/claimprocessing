# Metadata Filter Inference Fix Summary

## 1. Existing Issue

The metadata filter inference in `app/chains/agent_chain.py` was using overly simplistic keyword matching that incorrectly filtered health-related policy queries as motor insurance queries.

**Diagnostic Example:**
```json
{
  "query": "What is covered under day care procedures in this policy",
  "metadata_filter": {"insurance_type": "motor"},
  "result_count": 0
}
```

This is incorrect because "day care procedures" is a health insurance topic, not motor insurance. The issue occurred because the original logic only checked for basic keywords like "health", "medical", "motor", "car", "vehicle", causing false positives when "car" appeared as a substring in "day care".

## 2. Code Path Where Metadata Filter Was Created

**File:** `app/chains/agent_chain.py`

**Method:** `_handle_knowledge_retrieval` (lines 595-648)

**Original Logic (lines 605-616):**
```python
# Extract insurance type from message if mentioned
insurance_type = None
message_lower = message.lower()
if "health" in message_lower or "medical" in message_lower:
    insurance_type = "health"
elif "motor" in message_lower or "car" in message_lower or "vehicle" in message_lower:
    insurance_type = "motor"

# Build metadata filter if insurance type detected
metadata_filter = None
if insurance_type:
    metadata_filter = {"insurance_type": insurance_type}
```

## 3. Changes Made

### 3.1 New Method: `_infer_insurance_type`

Created a comprehensive keyword-based inference method that:

1. **Uses extensive keyword lists** for both health and motor insurance
2. **Orders keywords by specificity** (multi-word phrases first)
3. **Implements conflict handling** - returns `None` when both health and motor keywords are detected
4. **Returns `None` for unknown queries** - allows retrieval across all policies
5. **Includes defensive logging** at the metadata filter creation point

### 3.2 Updated `_handle_knowledge_retrieval` Method

Modified to use the new `_infer_insurance_type` method and added comprehensive logging:

```python
# Extract insurance type from message using improved inference
insurance_type = self._infer_insurance_type(message)

# Build metadata filter if insurance type detected
metadata_filter = None
if insurance_type:
    metadata_filter = {"insurance_type": insurance_type}

# Defensive logging at metadata filter creation point
logger.info(
    "Creating metadata filter",
    extra={
        "query": message,
        "detected_insurance_type": insurance_type,
        "metadata_filter": metadata_filter,
        "reason": f"Keyword-based detection: {insurance_type}" if insurance_type else "No clear insurance type detected",
    }
)
```

## 4. Health Keyword List

The following health-related keywords are used for detection (ordered by specificity):

### Multi-word Keywords (checked first):
- "day care procedure", "day care treatment"
- "pre-existing disease", "pre existing disease"
- "network hospital", "cashless claim"
- "room rent", "ambulance charges", "icu charges"
- "hospitalization", "hospitalisation"
- "ayush treatment", "new born", "organ donor"
- "health checkup", "medical expense", "medical bill"

### Single-word Keywords:
- "inpatient", "outpatient", "domiciliary", "maternity"
- "wellness", "preventive", "consultation", "diagnosis"
- "prescription", "therapy", "rehabilitation"
- "day care", "daycare"
- "health", "medical", "hospital", "surgery", "treatment"
- "disease", "illness", "icu", "ambulance", "cashless"
- "doctor", "patient", "opd"

## 5. Motor Keyword List

The following motor-related keywords are used for detection (ordered by specificity):

### Multi-word Keywords (checked first):
- "own damage", "third party", "third-party"
- "personal accident", "owner driver", "paid driver"
- "no claim bonus", "ncb"
- "accident damage", "total loss"
- "four wheeler", "two wheeler"
- "spare parts", "plate number"

### Single-word Keywords:
- "motor", "vehicle", "bike", "idv"
- "garage", "engine", "tyre", "bumper", "windshield"
- "collision", "theft", "comprehensive", "fuel"
- "registration", "rc", "chassis", "traffic", "road"
- "driving", "repair", "consumables", "depreciation", "premium"

**Note:** The keyword "car" was intentionally removed from the motor keyword list to avoid false substring matches with health terms like "day care" and "cardiac".

## 6. Conflict Handling Logic

When both health and motor keywords are detected in a single query:

1. **Log a warning** indicating conflicting signals
2. **Return `None`** for insurance_type
3. **Set metadata_filter to None** (no filtering)
4. **Retrieve across all policies** (both health and motor)

This ensures that ambiguous queries don't incorrectly filter out relevant results.

**Example conflicting queries:**
- "What is covered for motor vehicle accident with hospitalization?"
- "Is bike accident injury treatment covered?"
- "What is the coverage for vehicle accident with medical expenses?"

## 7. Unknown Handling Logic

When no insurance type is confidently detected:

1. **Log info** indicating no clear signal
2. **Return `None`** for insurance_type
3. **Set metadata_filter to None** (no filtering)
4. **Retrieve across all indexed chunks**

This ensures that generic queries like "What are the exclusions in this policy?" or "What is the policy term?" don't incorrectly filter results.

## 8. Validation Results

All 13 test queries passed successfully:

### Query 1: "What is covered under day care procedures in this policy?"
- **Expected:** insurance_type = health
- **Result:** ✅ PASSED
- **Detected keywords:** ['day care procedure', 'day care']

### Query 2: "What is covered under own damage in this policy?"
- **Expected:** insurance_type = motor
- **Result:** ✅ PASSED
- **Detected keywords:** ['own damage']

### Query 3: "What are the exclusions in this policy?"
- **Expected:** insurance_type = None
- **Result:** ✅ PASSED
- **Reason:** No clear insurance type detected

### Query 4: "What is covered for hospitalization?"
- **Expected:** insurance_type = health
- **Result:** ✅ PASSED
- **Detected keywords:** ['hospitalization', 'hospital']

### Query 5: "What is covered for vehicle accident damage?"
- **Expected:** insurance_type = motor
- **Result:** ✅ PASSED
- **Detected keywords:** ['vehicle', 'accident damage']

### Additional Test Coverage:
- ✅ Health keywords comprehensive (10 queries)
- ✅ Motor keywords comprehensive (10 queries)
- ✅ Conflicting keywords return None (3 queries)
- ✅ Unknown queries return None (4 queries)
- ✅ Case insensitive matching (4 queries)
- ✅ Metadata filter creation with logging
- ✅ Metadata filter None for ambiguous queries
- ✅ Logging contains required fields

## 9. Defensive Logging

Comprehensive logging was added at the metadata filter creation point:

### Inference Logging (INFO level):
```python
logger.info(
    "Metadata filter inference",
    extra={
        "query": message,
        "health_keywords_found": health_matches,
        "motor_keywords_found": motor_matches,
    }
)
```

### Conflict Warning (WARNING level):
```python
logger.warning(
    "Conflicting insurance type signals detected in query: %s. "
    "Health keywords: %s, Motor keywords: %s. "
    "Setting insurance_type to None to retrieve across all policies.",
    message,
    health_matches,
    motor_matches,
)
```

### Filter Creation Logging (INFO level):
```python
logger.info(
    "Creating metadata filter",
    extra={
        "query": message,
        "detected_insurance_type": insurance_type,
        "metadata_filter": metadata_filter,
        "reason": f"Keyword-based detection: {insurance_type}" if insurance_type else "No clear insurance type detected",
    }
)
```

## 10. Test Coverage

Created comprehensive test suite in `tests/test_metadata_filter_inference.py`:

- **13 test cases** covering all required scenarios
- **100% pass rate** on all tests
- Tests cover:
  - Day care procedures → health
  - Own damage → motor
  - Exclusions without context → None
  - Hospitalization → health
  - Vehicle accident damage → motor
  - Comprehensive health keywords
  - Comprehensive motor keywords
  - Conflicting keywords → None
  - Unknown queries → None
  - Case insensitive matching
  - Metadata filter creation with logging
  - Metadata filter None for ambiguous queries
  - Logging contains required fields

## 11. Impact

### Positive Impact:
- ✅ Health queries like "day care procedures" are now correctly identified as health insurance
- ✅ Motor queries are still correctly identified
- ✅ Ambiguous queries no longer cause incorrect filtering
- ✅ Unknown queries retrieve across all policies
- ✅ Comprehensive logging for debugging and monitoring

### No Breaking Changes:
- ✅ Retrieval ranking logic unchanged
- ✅ Vector index not rebuilt
- ✅ No modifications to unrelated files
- ✅ Backward compatible with existing functionality

## 12. Files Modified

1. **app/chains/agent_chain.py** - Added `_infer_insurance_type` method and updated `_handle_knowledge_retrieval`
2. **tests/test_metadata_filter_inference.py** - Created comprehensive test suite (13 tests)

## 13. Next Steps

The fix is complete and validated. The improved metadata filter inference will:

1. Correctly route health insurance queries (day care, hospitalization, surgery, etc.) to `insurance_type=health`
2. Correctly route motor insurance queries (own damage, IDV, engine, etc.) to `insurance_type=motor`
3. Return `None` for ambiguous or unknown queries, allowing retrieval across all policies
4. Provide comprehensive logging for monitoring and debugging