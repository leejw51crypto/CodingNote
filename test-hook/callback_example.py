def process_data(data, callback):
    """Process data and call the callback function with the result"""
    result = data * 2
    callback(result)


def print_result(value):
    """Callback function that prints the result"""
    print(f"Result: {value}")


def save_result(value):
    """Callback function that saves the result"""
    print(f"Saving {value} to file...")


def main():
    print("=== Callback Example ===")

    # Pass different callback functions
    process_data(5, print_result)
    process_data(10, save_result)

    # Use lambda as callback
    process_data(7, lambda x: print(f"Lambda result: {x}"))


if __name__ == "__main__":
    main()
