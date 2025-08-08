print("=== HOOK VS CALLBACK TIMING ===\n")

# HOOK EXAMPLE - FIXED TIMING
print("🔧 HOOK EXAMPLE:")
print("Step 1: Register hooks (timing is FIXED - will run when event triggers)")


def my_hook():
    print("   HOOK CALLED!")


hooks = []
hooks.append(my_hook)
print("   ✓ Hook registered")

print("Step 2: Do other work...")
print("   Working on something else...")

print("Step 3: Trigger event - hooks run automatically NOW!")
for hook in hooks:
    hook()  # FIXED timing - runs when WE trigger event

print("\n" + "=" * 40 + "\n")

# CALLBACK EXAMPLE - FLEXIBLE TIMING
print("📞 CALLBACK EXAMPLE:")
print("Step 1: Define callback function")


def my_callback():
    print("   CALLBACK CALLED!")


print("   ✓ Callback defined")

print("Step 2: Pass to function - RECEIVER decides when to call")


def early_caller(callback):
    print("   Early caller: calling callback RIGHT AWAY")
    callback()  # FLEXIBLE - caller decides timing


def late_caller(callback):
    print("   Late caller: doing work first...")
    print("   Late caller: still working...")
    print("   Late caller: NOW calling callback")
    callback()  # FLEXIBLE - different timing


early_caller(my_callback)
print()
late_caller(my_callback)

print("\n🎯 SUMMARY:")
print("Hook:     Timing FIXED when registered → runs when event triggers")
print("Callback: Timing FLEXIBLE → receiver decides when to call")
