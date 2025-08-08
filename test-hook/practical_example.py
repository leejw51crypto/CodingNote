class FileProcessor:
    def __init__(self):
        self.before_hooks = []
        self.after_hooks = []

    def add_before_hook(self, hook):
        """Add hook to run before processing"""
        self.before_hooks.append(hook)

    def add_after_hook(self, hook):
        """Add hook to run after processing"""
        self.after_hooks.append(hook)

    def process_file(self, filename, callback=None):
        """Process a file with hooks and optional callback"""
        # Run before hooks
        for hook in self.before_hooks:
            hook(filename)

        # Simulate file processing
        print(f"Processing {filename}...")
        result = f"Processed: {filename}"

        # Run after hooks
        for hook in self.after_hooks:
            hook(filename, result)

        # Call callback if provided
        if callback:
            callback(result)

        return result


def validate_file(filename):
    """Before hook: validate file"""
    print(f"✓ Validating {filename}")


def backup_file(filename):
    """Before hook: backup file"""
    print(f"✓ Creating backup of {filename}")


def log_completion(filename, result):
    """After hook: log completion"""
    print(f"✓ Logged completion of {filename}")


def send_notification(filename, result):
    """After hook: send notification"""
    print(f"✓ Sent notification: {filename} processed")


def save_to_db(result):
    """Callback: save result to database"""
    print(f"✓ Saved to database: {result}")


def main():
    print("=== Practical Example: File Processor ===")

    # Create processor
    processor = FileProcessor()

    # Register hooks
    processor.add_before_hook(validate_file)
    processor.add_before_hook(backup_file)
    processor.add_after_hook(log_completion)
    processor.add_after_hook(send_notification)

    # Process file with callback
    processor.process_file("document.txt", save_to_db)

    print("\n" + "=" * 40)

    # Process another file without callback
    processor.process_file("image.jpg")


if __name__ == "__main__":
    main()
