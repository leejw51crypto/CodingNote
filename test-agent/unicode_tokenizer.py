import random
import secrets
import tiktoken
from typing import List, Tuple


def analyze_tokenization(text: str) -> Tuple[List[int], List[str], int]:
    """
    Analyze how GPT tokenizes the given text using cl100k_base encoding
    Returns: (token_ids, token_strings, token_count)
    """
    # Get the cl100k_base encoding (used by GPT-4 and GPT-3.5-turbo)
    encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the text to get token IDs
    token_ids = encoding.encode(text)

    # Decode each token ID to see the actual token strings
    token_strings = [encoding.decode([token_id]) for token_id in token_ids]

    # Count tokens
    token_count = len(token_ids)

    return token_ids, token_strings, token_count


def display_tokenization_analysis(text: str, description: str = ""):
    """Display detailed tokenization analysis for given text"""
    print(f"\n{'='*70}")
    print(f"TOKENIZATION ANALYSIS: {description}")
    print(f"{'='*70}")
    print(f"Original text: {text}")
    print(f"Text length: {len(text)} characters")
    print(f"Byte length: {len(text.encode('utf-8'))} bytes")

    try:
        token_ids, token_strings, token_count = analyze_tokenization(text)

        print(f"\n🤖 Encoding: cl100k_base (GPT-4/GPT-3.5-turbo)")
        print(f"📊 Token count: {token_count}")
        print(f"💰 Compression ratio: {len(text)/token_count:.2f} chars per token")
        print(
            f"🗜️  Byte compression: {len(text.encode('utf-8'))/token_count:.2f} bytes per token"
        )

        print(f"\n🔤 Token breakdown:")
        for i, (token_id, token_str) in enumerate(zip(token_ids, token_strings)):
            # Show token with visual boundaries and escape special chars
            token_display = repr(token_str)
            byte_len = len(token_str.encode("utf-8"))
            print(f"  [{i+1:2d}] ID:{token_id:5d} ({byte_len:2d}b) → {token_display}")

        print(f"\n📝 Reconstructed: {''.join(token_strings)}")

    except Exception as e:
        print(f"❌ Error: {e}")


def analyze_languages():
    """Analyze tokenization of different languages"""

    languages = {
        "English": "Hello, how are you today?",
        "Spanish": "¡Hola! ¿Cómo estás hoy?",
        "French": "Bonjour ! Comment allez-vous aujourd'hui ?",
        "German": "Hallo! Wie geht es Ihnen heute?",
        "Chinese (Simplified)": "你好！你今天好吗？",
        "Chinese (Traditional)": "你好！你今天好嗎？",
        "Japanese": "こんにちは！今日はいかがですか？",
        "Korean": "안녕하세요! 오늘 어떠세요?",
        "Russian": "Привет! Как дела сегодня?",
        "Arabic": "مرحبا! كيف حالك اليوم؟",
        "Hindi": "नमस्ते! आज आप कैसे हैं?",
        "Thai": "สวัสดี! วันนี้เป็นอย่างไรบ้าง?",
        "Hebrew": "שלום! איך אתה היום?",
        "Greek": "Γεια σας! Πώς είστε σήμερα;",
    }

    print("🌍 UNICODE LANGUAGE TOKENIZATION ANALYSIS")
    print("=" * 70)
    print("This shows how GPT tokenizes different languages")
    print("Understanding cross-language tokenization is crucial for multilingual AI!\n")

    # Analyze each language
    for language, text in languages.items():
        display_tokenization_analysis(text, f"{language}")

    # Summary comparison
    print(f"\n{'='*70}")
    print("LANGUAGE EFFICIENCY COMPARISON")
    print(f"{'='*70}")
    print(
        f"{'Language':<20} | {'Chars':<5} | {'Bytes':<5} | {'Tokens':<6} | {'C/T':<4} | {'B/T':<4}"
    )
    print("-" * 70)

    for language, text in languages.items():
        _, _, token_count = analyze_tokenization(text)
        chars = len(text)
        bytes_len = len(text.encode("utf-8"))
        chars_per_token = chars / token_count
        bytes_per_token = bytes_len / token_count

        print(
            f"{language:<20} | {chars:<5} | {bytes_len:<5} | {token_count:<6} | {chars_per_token:<4.1f} | {bytes_per_token:<4.1f}"
        )


def analyze_emojis():
    """Analyze tokenization of emojis and special Unicode characters"""

    emoji_tests = {
        "Simple Emojis": "😀😂🥰😎🤔",
        "Complex Emojis": "👨‍💻👩‍🚀🧑‍🎨👨‍👩‍👧‍👦",
        "Flag Emojis": "🇺🇸🇯🇵🇩🇪🇫🇷🇨🇳",
        "Mixed Text + Emojis": "Hello! 😊 How are you? 🤔 Have a great day! 🌟",
        "Emoji Sequences": "🎉🎊🎈🎁🎂",
        "Skin Tone Modifiers": "👋🏻👋🏼👋🏽👋🏾👋🏿",
        "Gender Variants": "🧑‍💼👨‍💼👩‍💼",
        "Animal Emojis": "🐶🐱🐭🐹🐰🦊🐻🐼",
    }

    print(f"\n{'='*70}")
    print("EMOJI & SPECIAL UNICODE ANALYSIS")
    print(f"{'='*70}")

    for category, text in emoji_tests.items():
        display_tokenization_analysis(text, f"Emojis: {category}")


def analyze_special_unicode():
    """Analyze special Unicode characters and symbols"""

    special_tests = {
        "Mathematical Symbols": "∑∫∂∆∇√∞≈≠≤≥±×÷",
        "Currency Symbols": "$ € £ ¥ ₹ ₽ ₿ ¢ ₩ ₪",
        "Arrows": "→←↑↓↔↕⇒⇐⇑⇓⇔⇕",
        "Box Drawing": "┌─┐│└─┘┬┴├┤┼",
        "Braille": "⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵",
        "Diacritics": "àáâãäåæçèéêëìíîïñòóôõöøùúûüý",
        "Superscripts": "¹²³⁴⁵⁶⁷⁸⁹⁰ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ",
        "Subscripts": "₁₂₃₄₅₆₇₈₉₀ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ",
    }

    print(f"\n{'='*70}")
    print("SPECIAL UNICODE CHARACTERS ANALYSIS")
    print(f"{'='*70}")

    for category, text in special_tests.items():
        display_tokenization_analysis(text, f"Special: {category}")


def analyze_mixed_content():
    """Analyze mixed content with different Unicode ranges"""

    mixed_tests = {
        "Code with Unicode": "def hello_世界(): return '你好 World! 🌍'",
        "Multilingual Sentence": "Hello नमस्ते こんにちは 안녕하세요 مرحبا Здравствуйте",
        "Programming Symbols": "λ x: x² + √(x) ≈ ∫f(x)dx where x ∈ ℝ",
        "Social Media Text": "Just had an amazing lunch! 🍕🍟 #foodie #yummy 😋 @friend",
        "Scientific Notation": "E = mc² where c ≈ 3×10⁸ m/s and ℏ = 1.054×10⁻³⁴ J⋅s",
        "Mixed Scripts": "Русский English 中文 العربية हिंदी 日本語 한국어",
        "Web Content": "Visit https://example.com for more info! 📧 contact@example.com",
        "JSON with Unicode": '{"name": "张三", "emoji": "😊", "price": "¥100"}',
    }

    print(f"\n{'='*70}")
    print("MIXED UNICODE CONTENT ANALYSIS")
    print(f"{'='*70}")

    for category, text in mixed_tests.items():
        display_tokenization_analysis(text, f"Mixed: {category}")


def compare_encoding_efficiency():
    """Compare tokenization efficiency across different text types"""

    test_texts = {
        "ASCII Only": "The quick brown fox jumps over the lazy dog.",
        "Latin Extended": "Café naïve résumé piñata façade",
        "Cyrillic": "Быстрая коричневая лиса прыгает через ленивую собаку",
        "Chinese": "快速的棕色狐狸跳过懒惰的狗",
        "Japanese": "素早い茶色のキツネが怠け者の犬を飛び越える",
        "Arabic": "القفز السريع البني الثعلب فوق الكلب الكسول",
        "Emoji Heavy": "🦊🏃‍♂️💨🐕‍🦺😴 Quick fox jumps over lazy dog! 🌟✨",
        "Mixed Scripts": "The 快速 🦊 fox सर्दी over the 怠惰 🐕",
    }

    print(f"\n{'='*70}")
    print("ENCODING EFFICIENCY COMPARISON")
    print(f"{'='*70}")
    print(
        f"{'Text Type':<15} | {'Chars':<5} | {'Bytes':<5} | {'Tokens':<6} | {'C/T':<4} | {'B/T':<4} | {'Efficiency'}"
    )
    print("-" * 80)

    baseline_efficiency = None

    for text_type, text in test_texts.items():
        _, _, token_count = analyze_tokenization(text)
        chars = len(text)
        bytes_len = len(text.encode("utf-8"))
        chars_per_token = chars / token_count
        bytes_per_token = bytes_len / token_count

        if baseline_efficiency is None:
            baseline_efficiency = chars_per_token

        efficiency_ratio = chars_per_token / baseline_efficiency
        efficiency_desc = (
            "Excellent"
            if efficiency_ratio > 0.9
            else (
                "Good"
                if efficiency_ratio > 0.7
                else "Fair" if efficiency_ratio > 0.5 else "Poor"
            )
        )

        print(
            f"{text_type:<15} | {chars:<5} | {bytes_len:<5} | {token_count:<6} | {chars_per_token:<4.1f} | {bytes_per_token:<4.1f} | {efficiency_desc}"
        )


def interactive_unicode_tokenizer():
    """Interactive mode to analyze custom Unicode text"""
    print(f"\n{'='*70}")
    print("INTERACTIVE UNICODE TOKENIZER")
    print(f"{'='*70}")
    print("Enter any Unicode text to see how GPT tokenizes it!")
    print("Try different languages, emojis, symbols, mixed content...")
    print("Type 'quit' to exit, 'examples' to see sample texts")

    examples = [
        "你好世界! 🌍",
        "Здравствуй мир! 🌎",
        "مرحبا بالعالم! 🌏",
        "🚀 λx: x² + ∫f(x)dx ≈ ∑∞",
        "👨‍💻 def hello_世界(): return '🌟'",
    ]

    while True:
        user_input = input("\n🔤 Enter Unicode text (or 'quit'/'examples'): ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        elif user_input.lower() in ["examples", "ex", "e"]:
            print("\n📝 Example texts to try:")
            for i, example in enumerate(examples, 1):
                print(f"  {i}. {example}")
            continue
        elif user_input.isdigit() and 1 <= int(user_input) <= len(examples):
            text = examples[int(user_input) - 1]
            print(f"🎯 Using example: {text}")
            display_tokenization_analysis(text, f"Example #{user_input}")
        elif user_input:
            display_tokenization_analysis(user_input, "User Input")
        else:
            print("💡 Enter some text to analyze!")


def main():
    """Main function to run all demonstrations"""

    print("🌐 UNICODE TOKENIZATION DEMO")
    print(
        "This demonstrates how GPT tokenizes Unicode text across languages and scripts"
    )
    print(
        "Understanding Unicode tokenization is crucial for multilingual AI applications!\n"
    )

    # Run different analyses
    try:
        # 1. Language analysis
        analyze_languages()

        # 2. Emoji analysis
        analyze_emojis()

        # 3. Special Unicode characters
        analyze_special_unicode()

        # 4. Mixed content
        analyze_mixed_content()

        # 5. Efficiency comparison
        compare_encoding_efficiency()

        # 6. Interactive mode
        print(f"\n{'='*70}")
        choice = (
            input("Want to try the interactive Unicode tokenizer? (y/n): ")
            .lower()
            .strip()
        )
        if choice == "y":
            interactive_unicode_tokenizer()

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install required packages:")
        print("   pip install tiktoken")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
