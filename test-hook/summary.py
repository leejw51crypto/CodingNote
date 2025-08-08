"""
Callbacks (callback_example.py:1): Functions passed as arguments to be called later
- process_data() accepts a callback function
- Different callbacks can be passed for different behaviors

Hooks (hook_example.py:1): Event-driven pattern where multiple functions respond to events
- EventSystem manages hook registration and triggering
- Multiple hooks can respond to the same event

Practical Example (practical_example.py:1): Combines both patterns in a file processor
- Before/after hooks for validation, backup, logging
- Optional callback for final result processing

Key differences:
- Callback: One function called at a specific point
- Hook: Multiple functions triggered by an event
"""

print(
    "Callbacks (callback_example.py:1): Functions passed as arguments to be called later"
)
print("- process_data() accepts a callback function")
print("- Different callbacks can be passed for different behaviors")
print()
print(
    "Hooks (hook_example.py:1): Event-driven pattern where multiple functions respond to events"
)
print("- EventSystem manages hook registration and triggering")
print("- Multiple hooks can respond to the same event")
print()
print(
    "Practical Example (practical_example.py:1): Combines both patterns in a file processor"
)
print("- Before/after hooks for validation, backup, logging")
print("- Optional callback for final result processing")
print()
print("Key differences:")
print("- Callback: One function called at a specific point")
print("- Hook: Multiple functions triggered by an event")
