import time


class HookSystem:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, event, func):
        """Register hook - timing is FIXED when event triggers"""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(func)
        print(f"🔧 HOOK REGISTERED: {func.__name__} for '{event}' event")

    def trigger_event(self, event):
        """Trigger event - ALL hooks called automatically at this moment"""
        print(f"\n🚀 EVENT TRIGGERED: '{event}'")
        if event in self.hooks:
            for hook in self.hooks[event]:
                hook()


def flexible_function(callback=None):
    """Function that can call callback at different times"""
    print("📋 Starting work...")
    time.sleep(0.5)

    print("📋 Middle of work...")
    if callback:
        print("📋 Caller decides to call callback NOW")
        callback()  # FLEXIBLE timing - caller controls when

    time.sleep(0.5)
    print("📋 Work finished")


def hook_function():
    print("   ⚡ Hook executed!")


def callback_function():
    print("   📞 Callback executed!")


def main():
    print("=== TIMING DIFFERENCE EXAMPLE ===\n")

    print("1️⃣ HOOK - FIXED TIMING:")
    print("   Register hooks now, they'll be called when event triggers")

    hooks = HookSystem()
    # Timing is FIXED - hooks will run when 'start_work' event triggers
    hooks.register_hook("start_work", hook_function)
    hooks.register_hook("start_work", lambda: print("   ⚡ Another hook executed!"))

    print("\n   Doing other work...")
    time.sleep(1)

    print("   Now triggering event - ALL hooks run automatically:")
    hooks.trigger_event("start_work")  # FIXED timing - all hooks run now

    print("\n" + "=" * 50)

    print("\n2️⃣ CALLBACK - FLEXIBLE TIMING:")
    print("   Pass callback function, receiver controls WHEN to call it")

    # FLEXIBLE timing - flexible_function decides when to call callback
    flexible_function(callback_function)

    print("\n   Same callback, but called at different time:")

    def late_caller(callback):
        print("📋 Doing lots of work first...")
        time.sleep(1)
        print("📋 Finally ready to call callback")
        if callback:
            callback()  # Different timing!

    late_caller(callback_function)


if __name__ == "__main__":
    main()
