"""
Demo script to verify the policy status check functionality.
This demonstrates the fix for the issue where asking "can you verify if a claim is active?"
with policy P654321 was failing.
"""

from app.tools.policy_checker import check_policy_status

print("=" * 70)
print("POLICY STATUS CHECK DEMO")
print("=" * 70)

# Test the exact scenario from the task
print("\n1. Checking policy P654321 (the one from the task):")
print("-" * 70)
result = check_policy_status("P654321")
print(f"Policy Number: {result.policy_number}")
print(f"Status: {result.status}")
print(f"Is Active: {result.is_active}")
print(f"Message: {result.message}")
print(f"Details: {result.details}")

# Test an active policy for comparison
print("\n2. Checking policy P123456 (active policy for comparison):")
print("-" * 70)
result2 = check_policy_status("P123456")
print(f"Policy Number: {result2.policy_number}")
print(f"Status: {result2.status}")
print(f"Is Active: {result2.is_active}")
print(f"Message: {result2.message}")

# Test a non-existent policy
print("\n3. Checking non-existent policy P999999:")
print("-" * 70)
result3 = check_policy_status("P999999")
print(f"Policy Number: {result3.policy_number}")
print(f"Status: {result3.status}")
print(f"Is Active: {result3.is_active}")
print(f"Message: {result3.message}")

print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("\nThe system can now properly handle policy status queries like:")
print('  "Can you verify if a claim is active?"')
print('  "Is policy P654321 active?"')
print('  "Check the status of my policy"')