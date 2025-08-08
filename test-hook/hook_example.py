class EventSystem:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, event_name, hook_function):
        """Register a hook function for an event"""
        if event_name not in self.hooks:
            self.hooks[event_name] = []
        self.hooks[event_name].append(hook_function)

    def trigger_event(self, event_name, data=None):
        """Trigger all hooks registered for an event"""
        if event_name in self.hooks:
            for hook in self.hooks[event_name]:
                hook(data)


def on_user_login(user_data):
    """Hook function for user login"""
    print(f"User {user_data['name']} logged in!")


def send_welcome_email(user_data):
    """Hook function to send welcome email"""
    print(f"Sending welcome email to {user_data['email']}")


def log_login_event(user_data):
    """Hook function to log the event"""
    print(f"LOGIN EVENT: {user_data['name']} at {user_data.get('time', 'now')}")


def main():
    print("=== Hook Example ===")

    # Create event system
    events = EventSystem()

    # Register multiple hooks for the same event
    events.register_hook("user_login", on_user_login)
    events.register_hook("user_login", send_welcome_email)
    events.register_hook("user_login", log_login_event)

    # Trigger the event
    user = {"name": "Alice", "email": "alice@example.com", "time": "2025-08-08 10:30"}
    events.trigger_event("user_login", user)


if __name__ == "__main__":
    main()
