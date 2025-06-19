import os
import random
import re
from faker import Faker
from openai import OpenAI


def generate_fake_bids(num_bids=100, locale="en_US"):
    """Generate fake bids with names and amounts for house auction

    Args:
        num_bids (int): Number of bids to generate (default: 100)
        locale (str): Locale for name generation (default: 'en_US')
                     Options: 'en_US', 'en_GB', 'es_ES', 'fr_FR', 'de_DE', etc.
    """

    # Initialize Faker for generating realistic names
    fake = Faker(locale)

    bids = []
    used_names = set()

    for i in range(num_bids):
        # Generate unique name using Faker
        while True:
            full_name = fake.name()
            if full_name not in used_names:
                used_names.add(full_name)
                break

        # Generate bid amount between $200,000 and $800,000
        # Most bids in normal range, but a few outliers
        if i == 50:  # Make one clearly highest bid
            amount = random.randint(750000, 850000)
        elif i < 10:  # Some high bids
            amount = random.randint(600000, 750000)
        else:  # Regular bids
            amount = random.randint(200000, 650000)

        bids.append({"bidder_name": full_name, "amount": amount, "bid_number": i + 1})

    # Shuffle the list to randomize order
    random.shuffle(bids)

    return bids


def format_bids_for_ai(bids):
    """Format bids in a readable way for AI context"""
    formatted_text = "HOUSE AUCTION - ALL BIDS:\n"
    formatted_text += "=" * 50 + "\n"

    for bid in bids:
        formatted_text += (
            f"Bid #{bid['bid_number']}: {bid['bidder_name']} - ${bid['amount']:,}\n"
        )

    formatted_text += "=" * 50 + "\n"
    formatted_text += f"Total bids received: {len(bids)}"

    return formatted_text


def find_highest_bid_computed(bids):
    """Compute the actual highest bid"""
    highest_bid = max(bids, key=lambda x: x["amount"])
    return highest_bid


def chat_with_ai_about_auction(client, bids):
    """Send bids to AI and ask it to identify highest bidder"""

    # Format bids for AI
    bids_text = format_bids_for_ai(bids)

    # Initialize conversation for auction analysis
    conversation_history = [
        {
            "role": "system",
            "content": """You are an expert auction analyst with perfect attention to detail. Your job is to:

1. Carefully analyze ALL bid information provided
2. Identify the HIGHEST bidder by amount
3. Pay close attention to dollar amounts and compare them accurately
4. Provide the winner's name, bid number, and exact amount
5. Be absolutely certain of your analysis - double-check all numbers

When analyzing bids:
- Read through ALL bids carefully
- Compare amounts numerically (higher $ = higher bid)
- Identify the single highest amount
- Provide the corresponding bidder information

Your memory and context are crucial for this analysis!""",
        }
    ]

    # Send all bids to AI
    user_message = f"""{bids_text}

TASK: Please analyze all {len(bids)} bids above and identify the HIGHEST bidder.

Please provide:
1. Winner's name
2. Bid number  
3. Exact bid amount
4. Confirm this is the highest amount among all bids

Take your time to review all bids carefully before responding."""

    conversation_history.append({"role": "user", "content": user_message})

    # Show input being sent to AI
    print(f"\n{'='*60}")
    print("📤 INPUT TO AI:")
    print(f"{'='*60}")
    print("System Message:")
    print(conversation_history[0]["content"])
    print(f"\n{'-'*60}")
    print("User Message:")
    print(user_message)
    print(f"{'='*60}")

    try:
        # Get AI response
        response = client.chat.completions.create(
            # model="gpt-4-turbo",
            model="gpt-4o",
            messages=conversation_history,
            max_tokens=500,
            temperature=0.1,  # Lower temperature for more precise analysis
        )

        ai_response = response.choices[0].message.content

        # Show output from AI
        print(f"\n{'='*60}")
        print("📥 OUTPUT FROM AI:")
        print(f"{'='*60}")
        print(ai_response)
        print(f"{'='*60}")

        return ai_response

    except Exception as e:
        print(f"Error getting AI response: {e}")
        return None


def parse_ai_response_logically(ai_response, bids):
    """
    Logically parse AI response to extract winner information
    Returns dict with parsed information and confidence level
    """
    if not ai_response:
        return {
            "parsed_name": None,
            "parsed_amount": None,
            "parsed_bid_number": None,
            "confidence": 0,
            "parsing_method": "no_response",
            "issues": ["No AI response received"],
        }

    result = {
        "parsed_name": None,
        "parsed_amount": None,
        "parsed_bid_number": None,
        "confidence": 0,
        "parsing_method": "unknown",
        "issues": [],
    }

    # Create lookup dictionaries for validation
    valid_names = {bid["bidder_name"] for bid in bids}
    valid_amounts = {bid["amount"] for bid in bids}
    name_to_amount = {bid["bidder_name"]: bid["amount"] for bid in bids}
    amount_to_name = {bid["amount"]: bid["bidder_name"] for bid in bids}

    # Pattern 1: Look for explicit winner statements
    winner_patterns = [
        r"winner[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)",
        r"highest bidder[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)",
        r"([A-Z][a-z]+ [A-Z][a-z]+)\s*(?:is|has)\s*(?:the\s*)?(?:winner|highest)",
        r"1\.\s*(?:Winner\'s name|Name)[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)",
        r"Name[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)",
    ]

    # Pattern 2: Look for bid amounts
    amount_patterns = [
        r"\$(\d{1,3}(?:,\d{3})*)",
        r"(\d{1,3}(?:,\d{3})*)\s*dollars?",
        r"amount[:\s]*\$?(\d{1,3}(?:,\d{3})*)",
        r"bid[:\s]*\$?(\d{1,3}(?:,\d{3})*)",
    ]

    # Pattern 3: Look for bid numbers
    bid_number_patterns = [r"bid\s*#(\d+)", r"bid\s*number[:\s]*(\d+)", r"#(\d+):"]

    # Extract names
    extracted_names = []
    for pattern in winner_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        extracted_names.extend(matches)

    # Extract amounts
    extracted_amounts = []
    for pattern in amount_patterns:
        matches = re.findall(pattern, ai_response)
        for match in matches:
            try:
                # Remove commas and convert to int
                amount = int(match.replace(",", ""))
                extracted_amounts.append(amount)
            except ValueError:
                continue

    # Extract bid numbers
    extracted_bid_numbers = []
    for pattern in bid_number_patterns:
        matches = re.findall(pattern, ai_response, re.IGNORECASE)
        for match in matches:
            try:
                bid_num = int(match)
                extracted_bid_numbers.append(bid_num)
            except ValueError:
                continue

    # Validate and score extracted information
    valid_extracted_names = [name for name in extracted_names if name in valid_names]
    valid_extracted_amounts = [
        amount for amount in extracted_amounts if amount in valid_amounts
    ]

    # Determine most likely winner name
    if valid_extracted_names:
        # If multiple names, prefer the one that appears first or most frequently
        name_counts = {}
        for name in valid_extracted_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        result["parsed_name"] = max(name_counts.keys(), key=lambda x: name_counts[x])
        result["parsing_method"] = "regex_name_extraction"
    else:
        result["issues"].append("No valid winner name found")

    # Determine most likely winning amount
    if valid_extracted_amounts:
        # Prefer the highest amount mentioned (most likely to be the winning bid)
        result["parsed_amount"] = max(valid_extracted_amounts)
        result["parsing_method"] = "regex_amount_extraction"
    else:
        result["issues"].append("No valid bid amount found")

    # Cross-validate name and amount
    if result["parsed_name"] and result["parsed_amount"]:
        expected_amount = name_to_amount.get(result["parsed_name"])
        expected_name = amount_to_name.get(result["parsed_amount"])

        if expected_amount == result["parsed_amount"]:
            result["confidence"] = 95  # High confidence - name and amount match
            result["parsing_method"] += "_cross_validated"
        elif expected_name == result["parsed_name"]:
            result["confidence"] = 90  # High confidence - amount leads to correct name
            result["parsing_method"] += "_amount_validated"
        else:
            result["confidence"] = 60  # Medium confidence - mismatch detected
            result["issues"].append(
                f'Name-amount mismatch: {result["parsed_name"]} should have ${expected_amount:,}, but found ${result["parsed_amount"]:,}'
            )
    elif result["parsed_name"]:
        result["confidence"] = 70  # Medium confidence - name only
        result["parsed_amount"] = name_to_amount.get(result["parsed_name"])
    elif result["parsed_amount"]:
        result["confidence"] = 75  # Medium-high confidence - amount only
        result["parsed_name"] = amount_to_name.get(result["parsed_amount"])
    else:
        result["confidence"] = 10  # Very low confidence
        result["issues"].append("No valid winner information extracted")

    # Extract bid number if available
    if extracted_bid_numbers:
        result["parsed_bid_number"] = extracted_bid_numbers[0]  # Take first mentioned

    return result


def validate_ai_logic(ai_response, actual_highest, bids):
    """
    Comprehensive logical validation of AI response
    Returns detailed analysis of AI's reasoning and accuracy
    """
    parsing_result = parse_ai_response_logically(ai_response, bids)

    analysis = {
        "parsing_result": parsing_result,
        "accuracy_check": {
            "name_correct": False,
            "amount_correct": False,
            "bid_number_correct": False,
            "overall_correct": False,
        },
        "logic_check": {
            "found_valid_winner": False,
            "amount_is_maximum": False,
            "consistent_data": False,
            "reasoning_quality": "unknown",
        },
        "confidence_score": 0,
        "detailed_issues": [],
    }

    # Check accuracy
    if parsing_result["parsed_name"] == actual_highest["bidder_name"]:
        analysis["accuracy_check"]["name_correct"] = True
    else:
        analysis["detailed_issues"].append(
            f"Name mismatch: Expected '{actual_highest['bidder_name']}', got '{parsing_result['parsed_name']}'"
        )

    if parsing_result["parsed_amount"] == actual_highest["amount"]:
        analysis["accuracy_check"]["amount_correct"] = True
    else:
        analysis["detailed_issues"].append(
            f"Amount mismatch: Expected ${actual_highest['amount']:,}, got ${parsing_result['parsed_amount'] or 'None'}"
        )

    if parsing_result["parsed_bid_number"] == actual_highest["bid_number"]:
        analysis["accuracy_check"]["bid_number_correct"] = True

    analysis["accuracy_check"]["overall_correct"] = (
        analysis["accuracy_check"]["name_correct"]
        and analysis["accuracy_check"]["amount_correct"]
    )

    # Check logic
    if parsing_result["parsed_name"] and parsing_result["parsed_amount"]:
        analysis["logic_check"]["found_valid_winner"] = True

        # Check if the amount is actually the maximum
        all_amounts = [bid["amount"] for bid in bids]
        if parsing_result["parsed_amount"] == max(all_amounts):
            analysis["logic_check"]["amount_is_maximum"] = True
        else:
            analysis["detailed_issues"].append(
                f"AI selected amount ${parsing_result['parsed_amount']:,} but maximum is ${max(all_amounts):,}"
            )

        # Check data consistency
        expected_amount_for_name = next(
            (
                bid["amount"]
                for bid in bids
                if bid["bidder_name"] == parsing_result["parsed_name"]
            ),
            None,
        )
        if expected_amount_for_name == parsing_result["parsed_amount"]:
            analysis["logic_check"]["consistent_data"] = True
        else:
            analysis["detailed_issues"].append(
                f"Inconsistent data: {parsing_result['parsed_name']} should have ${expected_amount_for_name:,}"
            )

    # Assess reasoning quality
    reasoning_indicators = {
        "mentions_comparison": any(
            word in ai_response.lower()
            for word in ["highest", "maximum", "compare", "largest"]
        ),
        "shows_amount": parsing_result["parsed_amount"] is not None,
        "shows_confidence": any(
            word in ai_response.lower()
            for word in ["certain", "confident", "sure", "confirm"]
        ),
        "structured_response": bool(re.search(r"1\.|2\.|3\.", ai_response)),
    }

    reasoning_score = sum(reasoning_indicators.values())
    if reasoning_score >= 3:
        analysis["logic_check"]["reasoning_quality"] = "good"
    elif reasoning_score >= 2:
        analysis["logic_check"]["reasoning_quality"] = "fair"
    else:
        analysis["logic_check"]["reasoning_quality"] = "poor"

    # Calculate overall confidence
    accuracy_score = (
        sum(1 for v in analysis["accuracy_check"].values() if v) * 25
    )  # Max 100
    parsing_confidence = parsing_result["confidence"]
    logic_score = (
        sum(
            1
            for k, v in analysis["logic_check"].items()
            if k != "reasoning_quality" and v
        )
        * 20
    )  # Max 80

    analysis["confidence_score"] = min(
        100, (accuracy_score + parsing_confidence + logic_score) / 3
    )

    return analysis


def run_single_auction_test(client, iteration_num):
    """Run a single auction test and return results"""

    print(f"\n🏠 AUCTION TEST #{iteration_num}")
    print("=" * 50)

    # Generate bids
    bids = generate_fake_bids(100)

    # Compute actual highest bid
    actual_highest = find_highest_bid_computed(bids)
    print(
        f"💻 Computed Winner: {actual_highest['bidder_name']} - ${actual_highest['amount']:,}"
    )

    # Ask AI to analyze
    print(f"🤖 AI analyzing {len(bids)} bids...")
    ai_response = chat_with_ai_about_auction(client, bids)

    # Perform logical validation
    if ai_response:
        print("🔍 Performing logical analysis of AI response...")
        validation_result = validate_ai_logic(ai_response, actual_highest, bids)

        result = {
            "iteration": iteration_num,
            "actual_winner": actual_highest["bidder_name"],
            "actual_amount": actual_highest["amount"],
            "actual_bid_number": actual_highest["bid_number"],
            "ai_response": ai_response,
            "validation_result": validation_result,
            "ai_success": validation_result["accuracy_check"]["overall_correct"],
            "ai_got_amount": validation_result["accuracy_check"]["amount_correct"],
            "ai_got_name": validation_result["accuracy_check"]["name_correct"],
            "confidence_score": validation_result["confidence_score"],
            "parsing_method": validation_result["parsing_result"]["parsing_method"],
            "error": None,
        }

        # Enhanced result reporting
        if result["ai_success"]:
            print(
                f"✅ AI SUCCESS: Correctly identified highest bidder! (Confidence: {result['confidence_score']:.1f}%)"
            )
            print(f"   🎯 Method: {result['parsing_method']}")
        elif result["ai_got_amount"] or result["ai_got_name"]:
            correct_parts = []
            if result["ai_got_name"]:
                correct_parts.append("name")
            if result["ai_got_amount"]:
                correct_parts.append("amount")
            print(
                f"⚠️  AI PARTIAL: Got {' and '.join(correct_parts)} correct (Confidence: {result['confidence_score']:.1f}%)"
            )
            print(f"   🔍 Method: {result['parsing_method']}")
            if validation_result["detailed_issues"]:
                print(f"   ⚠️  Issues: {validation_result['detailed_issues'][0]}")
        else:
            print(
                f"❌ AI MISS: Did not identify correct highest bidder (Confidence: {result['confidence_score']:.1f}%)"
            )
            print(
                f"   🎯 Expected: {actual_highest['bidder_name']} - ${actual_highest['amount']:,}"
            )
            print(
                f"   🤖 AI Found: {validation_result['parsing_result']['parsed_name']} - ${validation_result['parsing_result']['parsed_amount'] or 'None'}"
            )
            if validation_result["detailed_issues"]:
                print(
                    f"   ❌ Issues: {'; '.join(validation_result['detailed_issues'][:2])}"
                )
    else:
        result = {
            "iteration": iteration_num,
            "actual_winner": actual_highest["bidder_name"],
            "actual_amount": actual_highest["amount"],
            "actual_bid_number": actual_highest["bid_number"],
            "ai_response": ai_response,
            "validation_result": None,
            "ai_success": False,
            "ai_got_amount": False,
            "ai_got_name": False,
            "confidence_score": 0,
            "parsing_method": "no_response",
            "error": "No AI response",
        }
        print("❌ AI ERROR: Failed to get response")

    return result


def run_multiple_auction_tests(num_iterations=5):
    """Run multiple auction tests and show statistics"""

    print("🏠 HOUSE AUCTION AI CONTEXT TEST - MULTIPLE ITERATIONS")
    print("=" * 70)
    print(f"Testing AI's context ability across {num_iterations} different auctions")
    print("Each auction has 100 unique bids with different winners\n")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return

    # Run tests
    results = []

    for i in range(1, num_iterations + 1):
        try:
            result = run_single_auction_test(client, i)
            results.append(result)

            # Brief pause between tests
            if i < num_iterations:
                print("⏳ Preparing next auction...")

        except Exception as e:
            print(f"❌ Error in iteration {i}: {e}")
            results.append(
                {
                    "iteration": i,
                    "error": str(e),
                    "ai_success": False,
                    "ai_got_amount": False,
                    "ai_got_name": False,
                }
            )

    # Display summary
    display_test_summary(results)

    return results


def display_test_summary(results):
    """Display comprehensive summary of all test results"""

    print(f"\n{'='*70}")
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print(f"{'='*70}")

    # Calculate statistics
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get("ai_success", False))
    partial_tests = sum(
        1
        for r in results
        if (r.get("ai_got_amount", False) or r.get("ai_got_name", False))
        and not r.get("ai_success", False)
    )
    failed_tests = sum(
        1
        for r in results
        if not r.get("ai_got_amount", False) and not r.get("ai_got_name", False)
    )
    error_tests = sum(1 for r in results if r.get("error"))

    # Calculate average confidence
    valid_results = [r for r in results if r.get("confidence_score", 0) > 0]
    avg_confidence = (
        sum(r.get("confidence_score", 0) for r in valid_results) / len(valid_results)
        if valid_results
        else 0
    )

    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    partial_rate = (partial_tests / total_tests) * 100 if total_tests > 0 else 0
    fail_rate = (failed_tests / total_tests) * 100 if total_tests > 0 else 0

    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Total Tests: {total_tests}")
    print(f"   ✅ Complete Success: {successful_tests} ({success_rate:.1f}%)")
    print(f"   ⚠️  Partial Success: {partial_tests} ({partial_rate:.1f}%)")
    print(f"   ❌ Failed: {failed_tests} ({fail_rate:.1f}%)")
    if error_tests > 0:
        print(f"   🚫 Errors: {error_tests}")
    print(f"   📈 Average Confidence: {avg_confidence:.1f}%")

    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 85)
    print(
        f"{'Test':<6} {'Winner Name':<20} {'Amount':<12} {'AI Result':<12} {'Confidence':<11} {'Method':<15}"
    )
    print("-" * 85)

    for result in results:
        if result.get("error"):
            status = "ERROR"
            winner = "N/A"
            amount = "N/A"
            confidence = "N/A"
            method = "N/A"
        else:
            if result.get("ai_success"):
                status = "✅ SUCCESS"
            elif result.get("ai_got_amount") or result.get("ai_got_name"):
                status = "⚠️ PARTIAL"
            else:
                status = "❌ FAILED"

            winner = result.get("actual_winner", "N/A")[:19]
            amount = f"${result.get('actual_amount', 0):,}"[:11]
            confidence = f"{result.get('confidence_score', 0):.1f}%"
            method = result.get("parsing_method", "unknown")[:14]

        print(
            f"#{result['iteration']:<5} {winner:<20} {amount:<12} {status:<12} {confidence:<11} {method:<15}"
        )

    print("-" * 85)

    # Parsing method analysis
    method_counts = {}
    if valid_results:
        print(f"\n🔍 PARSING METHOD ANALYSIS:")
        print("-" * 50)
        for result in valid_results:
            method = result.get("parsing_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        for method, count in sorted(
            method_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / len(valid_results)) * 100
            print(f"   {method}: {count} tests ({percentage:.1f}%)")

    # Show failed test details with logical analysis
    failed_results = [
        r for r in results if not r.get("ai_success", False) and not r.get("error")
    ]
    if failed_results:
        print(f"\n🔍 FAILED TEST LOGICAL ANALYSIS:")
        print("-" * 50)
        for result in failed_results:
            print(f"Test #{result['iteration']}:")
            print(f"  Expected Winner: {result.get('actual_winner', 'N/A')}")
            print(f"  Expected Amount: ${result.get('actual_amount', 0):,}")

            if result.get("validation_result"):
                val_result = result["validation_result"]
                parsing = val_result["parsing_result"]
                print(f"  AI Parsed Name: {parsing.get('parsed_name', 'None')}")
                print(
                    f"  AI Parsed Amount: ${parsing.get('parsed_amount', 0):,}"
                    if parsing.get("parsed_amount")
                    else "  AI Parsed Amount: None"
                )
                print(f"  Confidence: {result.get('confidence_score', 0):.1f}%")
                print(f"  Parsing Method: {result.get('parsing_method', 'unknown')}")

                # Show logic check results
                logic_check = val_result["logic_check"]
                print(f"  Logic Analysis:")
                print(
                    f"    - Found valid winner: {'✅' if logic_check['found_valid_winner'] else '❌'}"
                )
                print(
                    f"    - Amount is maximum: {'✅' if logic_check['amount_is_maximum'] else '❌'}"
                )
                print(
                    f"    - Data consistent: {'✅' if logic_check['consistent_data'] else '❌'}"
                )
                print(f"    - Reasoning quality: {logic_check['reasoning_quality']}")

                # Show detailed issues
                if val_result["detailed_issues"]:
                    print(f"  Issues: {val_result['detailed_issues'][0]}")
            print()

    # Enhanced context analysis with logical insights
    print(f"\n🧠 ENHANCED CONTEXT PERFORMANCE ANALYSIS:")
    if successful_tests > 7:
        print(
            "🌟 EXCELLENT: AI demonstrates strong context processing and logical reasoning"
        )
        print("   - Consistently identifies correct winners from 100+ bid context")
        print("   - Shows reliable numerical comparison and structured data extraction")
        print("   - Logical validation confirms high-quality reasoning")
    elif successful_tests > 5:
        print("👍 GOOD: AI shows decent context processing with logical analysis")
        print(
            "   - Generally handles large context well with occasional reasoning gaps"
        )
        print("   - Parsing methods show varied approaches to data extraction")
    elif successful_tests > 2:
        print("⚠️ FAIR: AI has mixed results with context processing and logic")
        print("   - Context handling and logical reasoning need improvement")
        print("   - Parsing confidence scores indicate uncertainty in analysis")
    else:
        print(
            "❌ POOR: AI struggles with large context processing and logical reasoning"
        )
        print("   - Significant difficulties in maintaining accuracy with complex data")
        print("   - Low parsing confidence suggests fundamental context issues")

    print(f"\n📈 LOGICAL ANALYSIS INSIGHTS:")
    if valid_results:
        print(
            f"   - Confidence scores range from {min(r.get('confidence_score', 0) for r in valid_results):.1f}% to {max(r.get('confidence_score', 0) for r in valid_results):.1f}%"
        )
        print(
            f"   - Average confidence: {avg_confidence:.1f}% indicates {'high' if avg_confidence > 80 else 'medium' if avg_confidence > 60 else 'low'} parsing reliability"
        )
        print(
            f"   - Parsing methods show {'diverse' if len(method_counts) > 2 else 'consistent'} extraction approaches"
        )
        print(
            f"   - Cross-validation {'frequently' if sum(1 for r in valid_results if 'validated' in r.get('parsing_method', '')) > len(valid_results)/2 else 'rarely'} confirms data consistency"
        )

    if partial_tests > 0:
        print(f"\n💡 PARTIAL SUCCESS LOGICAL ANALYSIS:")
        partial_results = [
            r
            for r in results
            if (r.get("ai_got_amount", False) or r.get("ai_got_name", False))
            and not r.get("ai_success", False)
        ]
        name_only = sum(
            1
            for r in partial_results
            if r.get("ai_got_name", False) and not r.get("ai_got_amount", False)
        )
        amount_only = sum(
            1
            for r in partial_results
            if r.get("ai_got_amount", False) and not r.get("ai_got_name", False)
        )

        print(f"   - {name_only} tests got name correct only")
        print(f"   - {amount_only} tests got amount correct only")
        print(
            f"   - Suggests {'name extraction' if name_only > amount_only else 'amount extraction' if amount_only > name_only else 'balanced'} parsing strengths/weaknesses"
        )

        # Analyze confidence for partial results
        partial_confidence = (
            sum(r.get("confidence_score", 0) for r in partial_results)
            / len(partial_results)
            if partial_results
            else 0
        )
        print(f"   - Partial results average confidence: {partial_confidence:.1f}%")


def run_auction_test():
    """Run the auction test (single or multiple)"""

    print("🏠 HOUSE AUCTION AI CONTEXT TEST")
    print("=" * 60)
    print("Choose test mode:")
    print("1. 🔄 Multiple Tests (10 iterations) - Recommended")
    print("2. 📝 Single Test (1 iteration)")
    print("3. 📊 Quick Demo (3 iterations)")

    while True:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3")

    if choice == "1":
        results = run_multiple_auction_tests(10)
    elif choice == "3":
        results = run_multiple_auction_tests(3)
    else:
        # Single test (original behavior)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is not set")
            print("Please set it using: export OPENAI_API_KEY='your-api-key'")
            return

        result = run_single_auction_test(client, 1)

        if result.get("ai_response"):
            print(f"\n🤖 AI DETAILED RESPONSE:")
            print("-" * 50)
            print(result["ai_response"])
            print("-" * 50)

    # Offer to show sample bid data
    if choice in ["1", "3"]:
        print(f"\n{'='*60}")
        show_sample = (
            input("Want to see a sample of the bid data structure? (y/n): ")
            .lower()
            .strip()
        )

        if show_sample == "y":
            print(f"\n📋 SAMPLE BID DATA (from last test):")
            print("=" * 50)
            sample_bids = generate_fake_bids(10)  # Generate small sample
            for bid in sample_bids[:10]:
                print(
                    f"Bid #{bid['bid_number']}: {bid['bidder_name']} - ${bid['amount']:,}"
                )
            print("(Each full test uses 100 bids like these)")


def run_legacy_single_test():
    """Original single test function for backward compatibility"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        return

    return run_single_auction_test(client, 1)


def main():
    """Main function to run auction test"""
    run_auction_test()


if __name__ == "__main__":
    main()
