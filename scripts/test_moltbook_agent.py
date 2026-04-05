"""
Test the Moltbook agent with extracted voice profile.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model.extraction.voice_extractor import VoiceExtractor
from world_model.models.agent import Tendency, AgentSet
from world_model.models.observation import Observation
from world_model.agents.moltbook_agent import (
    MoltbookAgent,
    MoltbookConfig,
    PromoTarget,
    DEFAULT_PROMO_TARGETS,
)


def main():
    # Load voice profile
    voice_path = Path(__file__).parent.parent / "data" / "twitter_extract" / "voice_profile.json"
    if not voice_path.exists():
        print(f"Voice profile not found at {voice_path}")
        print("Run scripts/extract_voice_only.py first")
        return

    extractor = VoiceExtractor()
    voice_profile = extractor.load_profile(str(voice_path))

    print("Loaded voice profile:")
    print(f"  Tone: {', '.join(voice_profile.tone_descriptors)}")
    print()

    # Create allocations reflecting the tweets (high MEANING, AUTONOMY, CURIOSITY)
    # These would normally come from trained World Model
    allocations = {
        Tendency.MEANING: 0.28,      # Civilizational paradigms, legacy
        Tendency.AUTONOMY: 0.25,     # Decentralization, freedom
        Tendency.CURIOSITY: 0.18,    # New ideas, exploration
        Tendency.CONNECTION: 0.10,   # Community building
        Tendency.STATUS: 0.08,       # Recognition
        Tendency.SURVIVAL: 0.06,     # Not primary driver
        Tendency.COMFORT: 0.05,      # Low priority
    }

    # Some sample observations (would come from extraction)
    observations = [
        Observation(content="Builds decentralized AI infrastructure"),
        Observation(content="Advocates for AI training and inference to be distributed"),
        Observation(content="Created on-chain jurisdiction frameworks"),
        Observation(content="Believes current systems are legacy software"),
        Observation(content="Uses psychedelics for psychological calibration"),
        Observation(content="Mixes philosophical depth with irreverent humor"),
        Observation(content="Bilingual in English and Romanian"),
        Observation(content="Critical of centralized power in tech"),
        Observation(content="Thinks about consciousness transfer to digital substrate"),
        Observation(content="Values freedom over salary stability"),
    ]

    # Create agent
    config = MoltbookConfig(
        name="Andrei Bot",
        posts_per_day=3,
        promo_frequency=0.3,
        promo_subtlety="high",
    )

    agent = MoltbookAgent(
        allocations=allocations,
        voice_profile=voice_profile,
        observations=observations,
        promo_targets=DEFAULT_PROMO_TARGETS,
        config=config,
    )

    # Generate some posts
    print("=" * 60)
    print("GENERATING TEST POSTS")
    print("=" * 60)

    for i in range(3):
        print(f"\n--- Post {i+1} ---")
        post = agent.generate_post()
        print(f"Tendency: {post.topic_tendency}")
        print(f"Promo: {post.promo_target or 'None'}")
        print(f"Content:\n{post.content}")

    # Test reply generation
    print("\n" + "=" * 60)
    print("TESTING REPLY GENERATION")
    print("=" * 60)

    test_posts = [
        "OpenAI just released a new model. Thoughts on centralized AI development?",
        "DAOs are failing because humans can't coordinate. Change my mind.",
        "What's the point of building anything when we're all gonna die anyway?",
    ]

    for post in test_posts:
        print(f"\n--- Replying to: '{post[:50]}...' ---")
        reply = agent.generate_reply(post)
        if reply:
            print(f"Content:\n{reply.content}")
        else:
            print("(No reply - not relevant to interests)")


if __name__ == "__main__":
    main()
