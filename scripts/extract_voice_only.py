"""Quick voice extraction from tweets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.extraction.tweet_processor import TweetProcessor
from world_model.extraction.voice_extractor import VoiceExtractor

def main():
    tweets_file = r"C:\Users\astmo\Downloads\twitter archive\tweets_text_only.txt"
    output_dir = Path(__file__).parent.parent / "data" / "twitter_extract"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing tweets...")
    processor = TweetProcessor()
    tweets = processor.process_file(tweets_file)

    tweet_texts = [t.content for t in tweets]
    print(f"Substantive tweets: {len(tweet_texts)}")

    print("\nExtracting voice profile (sampling 250 tweets)...")
    extractor = VoiceExtractor()
    profile = extractor.extract(tweet_texts, sample_size=250)

    output_path = output_dir / "voice_profile.json"
    extractor.save_profile(profile, str(output_path))

    print(f"\n{'='*60}")
    print("VOICE PROFILE")
    print(f"{'='*60}")
    print(f"\nTone: {', '.join(profile.tone_descriptors)}")
    print(f"\nRegister: {profile.register_range}")
    print(f"\nSentence patterns:")
    for p in profile.sentence_patterns:
        print(f"  - {p}")
    print(f"\nRhetorical devices:")
    for d in profile.rhetorical_devices:
        print(f"  - {d}")
    print(f"\nHumor patterns:")
    for h in profile.humor_patterns:
        print(f"  - {h}")
    print(f"\nConfrontation style: {profile.confrontation_style}")
    print(f"\nLanguage notes: {profile.language_notes}")
    print(f"\nExemplar tweets ({len(profile.exemplar_tweets)}):")
    for t in profile.exemplar_tweets[:10]:
        print(f'  "{t[:80]}..."' if len(t) > 80 else f'  "{t}"')

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
