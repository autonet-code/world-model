#!/usr/bin/env python3
"""
Extract atomic observations from a document.

Usage:
    python extract_observations.py <document_path> [--output <output_path>]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from world_model.models import Observation, ObservationStore, Source
from world_model.extraction.observation_extractor import ObservationExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract observations from document")
    parser.add_argument("document", help="Path to document to analyze")
    parser.add_argument("--output", "-o", default="observations.json", help="Output path")
    args = parser.parse_args()

    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Error: Document not found: {doc_path}")
        sys.exit(1)

    print(f"Reading document: {doc_path}")
    document = doc_path.read_text(encoding="utf-8")
    print(f"Document length: {len(document)} characters")

    source = Source(
        name=doc_path.stem,
        path=str(doc_path.absolute()),
        metadata={"type": "conversation_summary"}
    )

    print("Extracting observations via Claude CLI...")
    extractor = ObservationExtractor()
    store = ObservationStore()

    try:
        new_count, dup_count = extractor.extract_to_store(document, source, store)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Extracted {new_count} new observations ({dup_count} duplicates)")

    # Save
    output = {
        "source": source.name,
        "count": len(store),
        "observations": [obs.to_dict() for obs in store.all()]
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved to: {output_path}")

    # Print sample
    print("\nSample observations:")
    for obs in store.all()[:10]:
        print(f"  - {obs.content}")

    if len(store) > 10:
        print(f"  ... and {len(store) - 10} more")


if __name__ == "__main__":
    main()
