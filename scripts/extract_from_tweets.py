"""
Extract World Model observations and Voice Profile from Twitter archive.
Usage: python scripts/extract_from_tweets.py <tweets_text_only.txt>
"""

import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.extraction.tweet_processor import TweetProcessor
from world_model.extraction.voice_extractor import VoiceExtractor, VoiceProfile
from world_model.extraction.observation_extractor import ObservationExtractor
from world_model.models.observation import Observation, ObservationStore
from world_model.models.evidence import Source


def extract_voice(tweets: list[str], output_path: Path) -> VoiceProfile:
    """Extract voice profile from tweets."""
    print(f"\n{'='*60}")
    print("VOICE EXTRACTION")
    print(f"{'='*60}")
    print(f"Analyzing {len(tweets)} tweets for voice patterns...")

    extractor = VoiceExtractor()
    profile = extractor.extract(tweets, sample_size=250)

    extractor.save_profile(profile, str(output_path))
    print(f"Voice profile saved to: {output_path}")

    # Preview
    print("\n--- Voice Profile Preview ---")
    print(f"Tone: {', '.join(profile.tone_descriptors)}")
    print(f"Register: {profile.register_range}")
    print(f"Rhetorical devices: {', '.join(profile.rhetorical_devices[:5])}")
    print(f"Exemplars: {len(profile.exemplar_tweets)} tweets selected")

    return profile


def extract_observations(tweets: list[str], output_path: Path, batch_size: int = 50) -> ObservationStore:
    """Extract observations from tweets in batches."""
    print(f"\n{'='*60}")
    print("OBSERVATION EXTRACTION")
    print(f"{'='*60}")

    store = ObservationStore()
    extractor = ObservationExtractor()

    # Create batches
    batches = []
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i + batch_size]
        batches.append(batch)

    print(f"Processing {len(tweets)} tweets in {len(batches)} batches...")

    for i, batch in enumerate(batches):
        print(f"\nBatch {i+1}/{len(batches)} ({len(batch)} tweets)...")

        # Format batch as document
        document = "\n---\n".join(batch)

        source = Source(
            name=f"twitter_batch_{i+1}",
            metadata={"source_type": "twitter", "batch": i+1, "tweet_count": len(batch)}
        )

        try:
            new_count, dup_count = extractor.extract_to_store(document, source, store)
            print(f"  Extracted: {new_count} new, {dup_count} duplicates")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save observations
    obs_data = [obs.to_dict() for obs in store.all()]
    output_path.write_text(
        json.dumps(obs_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nTotal observations: {len(store)}")
    print(f"Saved to: {output_path}")

    return store


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_from_tweets.py <tweets_file>")
        print("       tweets_file: path to tweets_text_only.txt")
        sys.exit(1)

    tweets_file = Path(sys.argv[1])
    if not tweets_file.exists():
        print(f"File not found: {tweets_file}")
        sys.exit(1)

    # Output directory
    output_dir = Path(__file__).parent.parent / "data" / "twitter_extract"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process tweets
    print("Processing tweets...")
    processor = TweetProcessor()
    tweets = processor.process_file(str(tweets_file))

    print(f"\nProcessing stats:")
    for key, val in processor.stats.items():
        print(f"  {key}: {val}")

    # Get substantive tweet contents (English only for now, Romanian can be added)
    # Actually, keep both - the voice is bilingual
    tweet_texts = [t.content for t in tweets]

    print(f"\nSubstantive tweets: {len(tweet_texts)}")

    # Extract voice profile
    voice_profile = extract_voice(
        tweet_texts,
        output_dir / "voice_profile.json"
    )

    # Extract observations (this takes longer - batched API calls)
    print("\n" + "="*60)
    print("Starting observation extraction (this may take a while)...")
    print("="*60)

    # For initial run, let's do a subset to verify it works
    # Then can run full extraction
    sample_for_test = tweet_texts[:100]  # Start with 100 for testing

    observations = extract_observations(
        sample_for_test,  # Change to tweet_texts for full run
        output_dir / "observations_from_tweets.json",
        batch_size=25
    )

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Voice profile: {output_dir / 'voice_profile.json'}")
    print(f"Observations: {output_dir / 'observations_from_tweets.json'}")
    print(f"\nNext steps:")
    print("1. Review voice_profile.json")
    print("2. Run full observation extraction (change sample_for_test to tweet_texts)")
    print("3. Train World Model: python -m world_model.dynamics.trainer")


if __name__ == "__main__":
    main()
