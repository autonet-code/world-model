"""
End-to-end test: Novelty + Attention integration

This demonstrates:
1. Creating a reference frame with beliefs
2. Measuring novelty of incoming concepts
3. Showing how novelty scores would drive attention routing
"""

import sys
sys.path.insert(0, 'C:/code')

from world_model.novelty import measure_against_claims, Termination

# The agent's beliefs (reference frame)
BELIEFS = [
    "Traditional banking provides security and stability",
    "Trust in institutions is necessary for economic transactions",
    "Centralized systems are more efficient than decentralized ones",
    "Government regulation protects consumers",
]

# Incoming stream of concepts (simulating perception)
INCOMING = [
    "Bitcoin enables peer-to-peer transactions without intermediaries",
    "The Federal Reserve sets interest rates",
    "Photosynthesis converts sunlight to energy",
    "Blockchain eliminates the need for trusted third parties",
    "Banks offer savings accounts",
]

print("=" * 70)
print("END-TO-END TEST: Novelty -> Attention Routing")
print("=" * 70)
print()
print("Agent's beliefs (reference frame):")
for i, belief in enumerate(BELIEFS, 1):
    print(f"  {i}. {belief}")
print()
print("=" * 70)
print("Processing incoming stream...")
print("=" * 70)

results = []
for concept in INCOMING:
    print(f"\n>>> {concept[:50]}...")
    result = measure_against_claims(concept, BELIEFS, verbose=False)
    results.append((concept, result))

    # Attention routing decision
    if result.termination == Termination.CONTRADICTS_ROOT:
        route = "-> CONSCIOUS (contradicts beliefs!)"
        priority = "HIGH"
    elif result.termination == Termination.ORTHOGONAL:
        route = "-> discard (unrelated)"
        priority = "LOW"
    elif result.termination == Termination.INTEGRATED:
        route = "-> working_memory (familiar)"
        priority = "MEDIUM"
    else:
        route = "-> working_memory"
        priority = "MEDIUM"

    print(f"    Termination: {result.termination.value}")
    print(f"    Composite:   {result.composite:.3f}")
    print(f"    Routing:     {route}")

print()
print("=" * 70)
print("SUMMARY: What would reach conscious attention?")
print("=" * 70)

for concept, result in results:
    if result.termination == Termination.CONTRADICTS_ROOT:
        print(f"  HIGH PRIORITY: {concept[:60]}...")
print()
print("These items contradict the agent's beliefs and demand attention.")
