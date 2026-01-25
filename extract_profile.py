#!/usr/bin/env python3
"""
Extract a personal deviation profile from a document.

Usage:
    python extract_profile.py <document_path> [--output <output_path>]

Example:
    python extract_profile.py andrei_conversation_summary.md --output andrei_profile.json
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from world_model import DeviationExtractor, DeviationGraph, Source


def main():
    parser = argparse.ArgumentParser(description="Extract deviation profile from document")
    parser.add_argument("document", help="Path to document to analyze")
    parser.add_argument("--output", "-o", default="profile.json", help="Output path for profile JSON")
    parser.add_argument("--work-dir", "-w", default=None, help="Working directory for temp files")
    parser.add_argument("--summary", "-s", action="store_true", help="Print summary to stdout")
    args = parser.parse_args()

    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Error: Document not found: {doc_path}")
        sys.exit(1)

    print(f"Reading document: {doc_path}")
    document = doc_path.read_text(encoding="utf-8")
    print(f"Document length: {len(document)} characters")

    # Create source
    source = Source(
        name=doc_path.stem,
        path=str(doc_path.absolute()),
        metadata={"type": "conversation_summary"}
    )

    # Initialize extractor (uses Claude CLI) and graph
    print("Initializing extractor (using Claude CLI)...")
    extractor = DeviationExtractor(work_dir=args.work_dir)
    graph = DeviationGraph()
    graph.add_source(source)

    # Extract deviations
    print("Extracting deviations via Claude CLI...")
    print("(This may take a few minutes)")
    try:
        nodes = extractor.extract(document, source)
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Extracted {len(nodes)} deviation nodes")

    # Add to graph
    for node in nodes:
        graph.add_node(node)

    # Save
    output_path = Path(args.output)
    graph.save(str(output_path))
    print(f"Saved profile to: {output_path}")

    # Print summary if requested
    if args.summary:
        print("\n" + "=" * 60)
        print(graph.to_summary())
        print("\n" + "=" * 60)
        print("\nProfile excerpt:")
        print(graph.to_profile())


if __name__ == "__main__":
    main()
