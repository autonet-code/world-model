"""
World Model API Service

FastAPI service exposing world model operations.

Run locally:
    uvicorn api.main:app --reload

Endpoints:
    POST   /users/{user_id}/profile          - Create profile
    GET    /users/{user_id}/profile          - Get profile summary
    DELETE /users/{user_id}/profile          - Delete profile

    POST   /users/{user_id}/observations/extract  - Extract from document
    POST   /users/{user_id}/observations          - Add observations
    GET    /users/{user_id}/observations          - List observations

    POST   /users/{user_id}/trees                 - Create tree
    GET    /users/{user_id}/trees                 - List trees
    GET    /users/{user_id}/trees/{tree_id}       - Get tree
    POST   /users/{user_id}/trees/{tree_id}/stake - Stake observations
"""

import os
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from world_model import (
    WorldModel, ObservationStore, Observation, ObservationExtractor,
    AgentSet, Tree, HierarchicalStaker,
)
from world_model.storage import FirestoreAdapter
from world_model.models.evidence import Source


# ============================================================================
# Configuration
# ============================================================================

FIREBASE_CRED_PATH = os.environ.get("FIREBASE_CREDENTIALS", "serviceAccountKey.json")
USE_EMULATOR = os.environ.get("FIRESTORE_EMULATOR", "false").lower() == "true"

# Global adapter (initialized on startup)
db: Optional[FirestoreAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Firebase on startup."""
    global db
    try:
        if USE_EMULATOR:
            db = FirestoreAdapter.from_emulator()
            print("Connected to Firestore emulator")
        elif os.path.exists(FIREBASE_CRED_PATH):
            db = FirestoreAdapter.from_service_account(FIREBASE_CRED_PATH)
            print("Connected to Firestore")
        else:
            print(f"Warning: Firebase credentials not found at {FIREBASE_CRED_PATH}")
            print("Running without persistence (in-memory only)")
            db = None
    except Exception as e:
        print(f"Firebase init error: {e}")
        db = None
    yield


app = FastAPI(
    title="World Model API",
    description="API for managing personal world models",
    version="0.4.0",
    lifespan=lifespan,
)

# CORS for Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateProfileRequest(BaseModel):
    name: str
    metadata: Optional[dict] = None


class ProfileResponse(BaseModel):
    name: str
    created_at: datetime
    updated_at: datetime
    observation_count: int
    tree_count: int
    agents: dict
    trees: list[dict]


class ExtractRequest(BaseModel):
    document: str
    source_name: str
    source_metadata: Optional[dict] = None


class AddObservationsRequest(BaseModel):
    observations: list[str]
    source_name: Optional[str] = "api"


class ObservationResponse(BaseModel):
    id: str
    content: str
    source_id: str
    timestamp: datetime


class CreateTreeRequest(BaseModel):
    root_value: str
    description: Optional[str] = ""


class TreeResponse(BaseModel):
    id: str
    root_value: str
    description: str
    score: float
    depth: int
    node_count: int


class TreeDetailResponse(TreeResponse):
    root_node: dict


class StakeRequest(BaseModel):
    observation_ids: Optional[list[str]] = None  # None = all observations
    anchor_threshold: float = 0.6


class StakeResponse(BaseModel):
    anchors: int
    relational_nodes: int
    skipped: int
    tree_score: float
    tree_depth: int


# ============================================================================
# In-memory fallback when no Firestore
# ============================================================================

_memory_store: dict[str, WorldModel] = {}


def get_model(user_id: str) -> Optional[WorldModel]:
    """Get world model from Firestore or memory."""
    if db:
        return db.load_world_model(user_id)
    return _memory_store.get(user_id)


def save_model(user_id: str, model: WorldModel) -> None:
    """Save world model to Firestore or memory."""
    if db:
        db.save_world_model(user_id, model)
    else:
        _memory_store[user_id] = model


# ============================================================================
# Profile Endpoints
# ============================================================================

@app.post("/users/{user_id}/profile", response_model=ProfileResponse)
async def create_profile(user_id: str, request: CreateProfileRequest):
    """Create a new user profile / world model."""
    existing = get_model(user_id)
    if existing:
        raise HTTPException(400, "Profile already exists")

    model = WorldModel(
        name=request.name,
        metadata=request.metadata or {},
    )
    save_model(user_id, model)

    return ProfileResponse(
        name=model.name,
        created_at=model.created_at,
        updated_at=model.updated_at,
        observation_count=len(model.observations),
        tree_count=len(model.trees),
        agents=model.agents.to_dict(),
        trees=[],
    )


@app.get("/users/{user_id}/profile", response_model=ProfileResponse)
async def get_profile(user_id: str):
    """Get user profile summary."""
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    return ProfileResponse(
        name=model.name,
        created_at=model.created_at,
        updated_at=model.updated_at,
        observation_count=len(model.observations),
        tree_count=len(model.trees),
        agents=model.agents.to_dict(),
        trees=[
            {
                "id": t.id,
                "root_value": t.root_value,
                "score": t.score,
                "depth": t.depth(),
            }
            for t in model.trees.all()
        ],
    )


@app.delete("/users/{user_id}/profile")
async def delete_profile(user_id: str):
    """Delete user profile and all data."""
    if db:
        db.delete_world_model(user_id)
    elif user_id in _memory_store:
        del _memory_store[user_id]
    else:
        raise HTTPException(404, "Profile not found")

    return {"status": "deleted"}


# ============================================================================
# Observation Endpoints
# ============================================================================

@app.post("/users/{user_id}/observations/extract")
async def extract_observations(
    user_id: str,
    request: ExtractRequest,
    background_tasks: BackgroundTasks,
):
    """
    Extract observations from a document.

    This calls Claude to parse the document into atomic observations.
    Can be slow for large documents.
    """
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    source = Source(
        name=request.source_name,
        path="api",
        metadata=request.source_metadata or {},
    )

    extractor = ObservationExtractor()
    new_count, dup_count = extractor.extract_to_store(
        request.document,
        source,
        model.observations,
    )

    save_model(user_id, model)

    return {
        "new_observations": new_count,
        "duplicates": dup_count,
        "total_observations": len(model.observations),
    }


@app.post("/users/{user_id}/observations")
async def add_observations(user_id: str, request: AddObservationsRequest):
    """Add observations directly (without extraction)."""
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    added = 0
    for content in request.observations:
        obs = Observation(
            content=content,
            source_id=request.source_name or "api",
        )
        _, is_new = model.observations.add(obs)
        if is_new:
            added += 1

    save_model(user_id, model)

    return {
        "added": added,
        "total_observations": len(model.observations),
    }


@app.get("/users/{user_id}/observations", response_model=list[ObservationResponse])
async def list_observations(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
):
    """List observations."""
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    observations = model.observations.all()[offset:offset + limit]

    return [
        ObservationResponse(
            id=obs.id,
            content=obs.content,
            source_id=obs.source_id,
            timestamp=obs.timestamp,
        )
        for obs in observations
    ]


# ============================================================================
# Tree Endpoints
# ============================================================================

@app.post("/users/{user_id}/trees", response_model=TreeResponse)
async def create_tree(user_id: str, request: CreateTreeRequest):
    """Create a new value tree."""
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    tree = Tree(
        root_value=request.root_value,
        description=request.description or "",
    )
    model.trees.add(tree)
    save_model(user_id, model)

    return TreeResponse(
        id=tree.id,
        root_value=tree.root_value,
        description=tree.description,
        score=tree.score,
        depth=tree.depth(),
        node_count=len(tree.all_nodes()),
    )


@app.get("/users/{user_id}/trees", response_model=list[TreeResponse])
async def list_trees(user_id: str):
    """List all trees."""
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    return [
        TreeResponse(
            id=t.id,
            root_value=t.root_value,
            description=t.description,
            score=t.score,
            depth=t.depth(),
            node_count=len(t.all_nodes()),
        )
        for t in model.trees.all()
    ]


@app.get("/users/{user_id}/trees/{tree_id}", response_model=TreeDetailResponse)
async def get_tree(user_id: str, tree_id: str):
    """Get tree with full structure."""
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    tree = model.trees.get(tree_id)
    if not tree:
        raise HTTPException(404, "Tree not found")

    return TreeDetailResponse(
        id=tree.id,
        root_value=tree.root_value,
        description=tree.description,
        score=tree.score,
        depth=tree.depth(),
        node_count=len(tree.all_nodes()),
        root_node=tree.root_node.to_dict() if tree.root_node else {},
    )


@app.post("/users/{user_id}/trees/{tree_id}/stake", response_model=StakeResponse)
async def stake_tree(user_id: str, tree_id: str, request: StakeRequest):
    """
    Stake observations into a tree using hierarchical staking.

    This calls Claude multiple times - can be slow.
    """
    model = get_model(user_id)
    if not model:
        raise HTTPException(404, "Profile not found")

    tree = model.trees.get(tree_id)
    if not tree:
        raise HTTPException(404, "Tree not found")

    # Get observations to stake
    if request.observation_ids:
        observations = [
            model.observations.get(oid)
            for oid in request.observation_ids
            if model.observations.get(oid)
        ]
    else:
        observations = model.observations.all()

    if not observations:
        raise HTTPException(400, "No observations to stake")

    # Create temp store for staking
    store = ObservationStore()
    for obs in observations:
        store.add(obs)

    # Stake
    staker = HierarchicalStaker()
    stats = staker.stake_all(
        store,
        tree,
        model.agents,
        anchor_threshold=request.anchor_threshold,
        verbose=False,
    )

    save_model(user_id, model)

    return StakeResponse(
        anchors=stats["anchors"],
        relational_nodes=stats["relational_nodes"],
        skipped=stats["skipped"],
        tree_score=stats["tree_score"],
        tree_depth=stats["tree_depth"],
    )


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "firestore": "connected" if db else "not configured",
        "version": "0.4.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
