#!/usr/bin/env python3
"""
Test script to demonstrate Faker name generation with different locales
"""

from auction import generate_fake_bids


def test_different_locales():
    """Test name generation with different locales"""

    locales = {
        "en_US": "English (US)",
        "en_GB": "English (UK)",
        "es_ES": "Spanish (Spain)",
        "fr_FR": "French (France)",
        "de_DE": "German (Germany)",
        "it_IT": "Italian (Italy)",
        "ja_JP": "Japanese (Japan)",
        "ko_KR": "Korean (South Korea)",
    }

    print("🌍 FAKER NAME GENERATION TEST - DIFFERENT LOCALES")
    print("=" * 60)

    for locale_code, locale_name in locales.items():
        print(f"\n📍 {locale_name} ({locale_code}):")
        print("-" * 40)

        try:
            # Generate 5 bids for each locale
            bids = generate_fake_bids(5, locale=locale_code)

            for bid in bids:
                print(f"  {bid['bidder_name']} - ${bid['amount']:,}")

        except Exception as e:
            print(f"  ❌ Error with {locale_code}: {e}")

    print(f"\n{'=' * 60}")
    print("✅ Faker name generation test completed!")
    print("💡 The default auction system now uses realistic names from Faker")


if __name__ == "__main__":
    test_different_locales()
