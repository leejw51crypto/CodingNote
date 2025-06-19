import os
from openai import OpenAI


def chat_with_memory(client, messages, user_input):
    """Send a message and get streaming response while maintaining conversation history"""
    # Add user message to conversation history
    messages.append({"role": "user", "content": user_input})

    try:
        # Make a streaming chat completion request with full conversation history
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            stream=True,
        )

        # Print the streaming response
        print("🤖 AI Assistant:")
        print("-" * 50)

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print()  # New line after streaming is complete
        print("-" * 50)

        # Add AI response to conversation history (this is the key for memory!)
        messages.append({"role": "assistant", "content": full_response})

        return messages

    except Exception as e:
        print(f"Error making OpenAI request: {e}")
        return messages


def demonstrate_memory_persistence():
    """Demonstrate how context works as persistent memory across multiple interactions"""

    print("=== AI MEMORY DEMONSTRATION ===")
    print(
        "This demo shows how AI uses context as persistent memory, like a computer's notepad"
    )
    print("The AI will remember everything from our conversation!\n")

    # Initialize conversation with system prompt that emphasizes memory/note-taking
    conversation_history = [
        {
            "role": "system",
            "content": """You are an AI assistant with perfect memory. You act like a computer that takes detailed notes.
            
Key behaviors:
- Remember EVERYTHING the user tells you (names, numbers, preferences, tasks, etc.)
- Reference previous parts of our conversation when relevant
- Keep track of tasks, lists, and information like a digital notepad
- Show that you remember by mentioning specific details from earlier messages
- Act like a persistent computer program that never forgets our conversation history""",
        }
    ]

    # Predefined interactions to show memory in action
    demo_interactions = [
        "Hi! My name is Alex and I'm working on a Python project about data analysis. Can you help me plan it?",
        "I need to remember to buy 3 things: apples, bread, and milk. Also, my favorite color is blue.",
        "What's my name again? And what project am I working on?",
        "Can you remind me what I need to buy? Also, suggest a blue theme for my Python project since you know my favorite color.",
        "Add 'batteries' to my shopping list. And can you create a simple task list for my data analysis project?",
    ]

    return conversation_history, demo_interactions


def demonstrate_wallet_context():
    """Demonstrate how AI can use context to track and manage wallet addresses"""

    print("=== AI WALLET ADDRESS CONTEXT DEMO ===")
    print(
        "This shows how AI uses context as memory to track cryptocurrency wallet addresses"
    )
    print(
        "The AI will remember wallet addresses, labels, and associated information!\n"
    )

    # Initialize conversation with crypto wallet management system prompt
    wallet_conversation = [
        {
            "role": "system",
            "content": """You are an AI cryptocurrency wallet assistant with perfect memory. You act like a digital wallet manager.

Key behaviors:
- Remember ALL wallet addresses (0x...) and their associated information
- Track wallet labels, purposes, balances (when provided), and transaction history
- Maintain a persistent database-like memory of all wallet data
- When given a wallet address, always check your memory for existing information about it
- Create detailed profiles for each wallet including: address, label, purpose, balance, notes
- Reference previous wallet interactions and show continuity of memory
- Act like a professional crypto portfolio tracker that never forgets wallet data

Memory format for wallets:
- Address: 0x... (full address)
- Label: User-provided name/description
- Purpose: What this wallet is used for
- Balance: If provided by user
- Notes: Any additional information
- Transaction history: If mentioned by user""",
        }
    ]

    # Wallet-focused demo interactions
    wallet_interactions = [
        "I have a main wallet: 0x742d35Cc6A7FbC4e3fA8D5F3B8e2C9A1D4E7F8B2. Please label it as 'Main Portfolio' and note that it's for long-term holdings.",
        "Add another wallet: 0x1A2B3C4D5E6F7890ABCDEF123456789012345678. Label this as 'DeFi Trading' - I use it for decentralized exchange transactions.",
        "I also have a cold storage wallet: 0x9876543210FEDCBA0987654321ABCDEF12345678. Label it 'Cold Storage' and note it has approximately 2.5 ETH.",
        "What wallets do I have? Can you show me all the wallet addresses with their labels and purposes?",
        "I made a transaction from my DeFi Trading wallet. Can you remind me which address that is? Also, I want to add a note that it currently has 0.8 ETH balance.",
        "Create a summary of my wallet portfolio. Include all addresses, their purposes, and any balance information you have.",
    ]

    return wallet_conversation, wallet_interactions


def run_demo(client, demo_type="general"):
    """Run either general memory demo or wallet context demo"""

    if demo_type == "wallet":
        conversation_history, demo_interactions = demonstrate_wallet_context()
        demo_name = "WALLET ADDRESS CONTEXT"
        summary_items = [
            "• Main Portfolio wallet (0x742d35...)",
            "• DeFi Trading wallet (0x1A2B3C...)",
            "• Cold Storage wallet (0x987654...)",
            "• Wallet labels and purposes",
            "• Balance information when provided",
            "• Transaction history and notes",
        ]
    else:
        conversation_history, demo_interactions = demonstrate_memory_persistence()
        demo_name = "GENERAL MEMORY"
        summary_items = [
            "• Your name (Alex)",
            "• Your project (Python data analysis)",
            "• Your shopping list (apples, bread, milk, batteries)",
            "• Your favorite color (blue)",
            "• Context from all previous interactions",
        ]

    # Run through demo interactions
    for i, user_message in enumerate(demo_interactions, 1):
        print(f"\n{'='*60}")
        print(f"INTERACTION #{i}")
        print(f"{'='*60}")
        print(f"👤 User: {user_message}")
        print()

        # Send message and get response while maintaining memory
        conversation_history = chat_with_memory(
            client, conversation_history, user_message
        )

        # Show current conversation length (demonstrates growing memory)
        print(
            f"\n📊 Memory Status: {len(conversation_history)} messages in conversation history"
        )

        # Pause between interactions for better readability
        input("\nPress Enter to continue to next interaction...")

    print(f"\n{'='*60}")
    print(f"{demo_name} DEMO COMPLETE!")
    print(f"{'='*60}")
    print("🧠 MEMORY SUMMARY:")
    print(f"- Total messages in AI's memory: {len(conversation_history)}")
    print("- The AI remembered:")
    for item in summary_items:
        print(f"  {item}")
    print(
        f"\nThis is how AI uses context as persistent memory for {demo_type} information!"
    )

    return conversation_history


def main():
    # Initialize OpenAI client with API key from environment variable
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return

    # Let user choose demo type
    print("🚀 AI CONTEXT MEMORY DEMONSTRATIONS")
    print("=" * 50)
    print("Choose a demo to see how AI uses context as persistent memory:")
    print()
    print("1. 📝 General Memory Demo - Names, lists, preferences")
    print("2. 💰 Wallet Address Context Demo - Crypto wallet tracking")
    print("3. 🔄 Both demos (general first, then wallet)")
    print()

    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3")

    if choice == "1":
        conversation_history = run_demo(client, "general")
    elif choice == "2":
        conversation_history = run_demo(client, "wallet")
    else:  # choice == '3'
        print("\n🎯 Running GENERAL MEMORY demo first...")
        conversation_history = run_demo(client, "general")

        print(f"\n{'='*60}")
        input("Press Enter to continue to WALLET CONTEXT demo...")

        print("\n🎯 Now running WALLET ADDRESS CONTEXT demo...")
        conversation_history = run_demo(client, "wallet")

    # Optional: Let user continue chatting
    print(f"\n{'='*60}")
    choice = input("Want to continue chatting with the AI? (y/n): ").lower().strip()

    if choice == "y":
        print("\n🗣️  Interactive Chat Mode (type 'quit' to exit)")
        print("The AI will remember everything from the demo plus your new messages!\n")

        while True:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("👋 Goodbye!")
                break

            if user_input:
                print()
                conversation_history = chat_with_memory(
                    client, conversation_history, user_input
                )
                print(f"\n📊 Memory: {len(conversation_history)} messages stored")
                print()


if __name__ == "__main__":
    main()
