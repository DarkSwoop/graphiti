"""
Standalone FastAPI app that exposes a /benchmark endpoint.
Runs INSIDE the Graphiti container alongside the main app.
Accepts search config as parameter to test different strategies.
"""
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from graphiti_core.search.search_config import (
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)

from graph_service.zep_graphiti import create_configured_graphiti, build_advanced_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
graphiti_client = None


# Pre-defined search configs to benchmark
CONFIGS = {
    "edge_rrf": SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        limit=10,
    ),
    "edge_mmr": SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.mmr,
            mmr_lambda=0.7,
        ),
        limit=10,
    ),
    "edge_episode_mentions": SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.episode_mentions,
        ),
        limit=10,
    ),
    "combined_rrf": SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.rrf,
        ),
        community_config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.rrf,
        ),
        limit=10,
    ),
    "combined_mmr": SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.mmr,
            mmr_lambda=0.7,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.mmr,
            mmr_lambda=0.7,
        ),
        community_config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.mmr,
            mmr_lambda=0.7,
        ),
        limit=10,
    ),
    "combined_rrf_with_bfs": SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity, EdgeSearchMethod.bfs],
            reranker=EdgeReranker.rrf,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity, NodeSearchMethod.bfs],
            reranker=NodeReranker.rrf,
        ),
        community_config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.rrf,
        ),
        limit=10,
    ),
    "combined_rrf_with_episodes": SearchConfig(
        edge_config=EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.rrf,
        ),
        node_config=NodeSearchConfig(
            search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
            reranker=NodeReranker.rrf,
        ),
        episode_config=EpisodeSearchConfig(
            search_methods=[EpisodeSearchMethod.bm25],
            reranker=EpisodeReranker.rrf,
        ),
        community_config=CommunitySearchConfig(
            search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            reranker=CommunityReranker.rrf,
        ),
        limit=10,
    ),
}


class BenchmarkQuery(BaseModel):
    query: str
    group_ids: list[str] = Field(default=["retrieval-test"])
    config_name: str = Field(default="edge_rrf", description="One of: " + ", ".join(CONFIGS.keys()))
    exclude_expired: bool = True
    max_results: int = 10


class BenchmarkAllQuery(BaseModel):
    query: str
    group_ids: list[str] = Field(default=["retrieval-test"])
    exclude_expired: bool = True
    max_results: int = 10


@app.on_event("startup")
async def startup():
    global graphiti_client
    graphiti_client = await create_configured_graphiti()
    logger.info("Benchmark graphiti client ready")


@app.on_event("shutdown")
async def shutdown():
    if graphiti_client:
        await graphiti_client.close()


@app.get("/health")
async def health():
    return {"status": "ok", "configs": list(CONFIGS.keys())}


@app.post("/benchmark")
async def benchmark(req: BenchmarkQuery):
    """Run a single query with a specific search config."""
    config = CONFIGS.get(req.config_name)
    if not config:
        return {"error": f"Unknown config: {req.config_name}", "available": list(CONFIGS.keys())}

    config_copy = config.model_copy()
    config_copy.limit = req.max_results

    search_filter = SearchFilters()
    if req.exclude_expired:
        search_filter = SearchFilters(
            expired_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]]
        )

    import time
    start = time.time()

    results = await graphiti_client.search_(
        query=req.query,
        config=config_copy,
        group_ids=req.group_ids,
        search_filter=search_filter,
    )

    await graphiti_client.resolve_missing_node_names(results)
    enriched = build_advanced_results(results)

    latency_ms = (time.time() - start) * 1000

    return {
        "config": req.config_name,
        "query": req.query,
        "latency_ms": round(latency_ms, 1),
        "facts": [
            {
                "fact": f.fact,
                "source": f.source_entity,
                "target": f.target_entity,
                "score": round(f.score, 4),
                "name": f.name,
            }
            for f in enriched.facts
        ],
        "entities": [
            {"name": e.name, "labels": e.labels, "summary": e.summary[:100] if e.summary else ""}
            for e in enriched.entities
        ],
        "communities": [
            {"name": c.name, "summary": c.summary[:100] if c.summary else ""}
            for c in enriched.communities
        ],
    }


@app.post("/benchmark-all")
async def benchmark_all(req: BenchmarkAllQuery):
    """Run a query against ALL configs and return comparison."""
    results = {}
    for config_name in CONFIGS:
        try:
            single = await benchmark(BenchmarkQuery(
                query=req.query,
                group_ids=req.group_ids,
                config_name=config_name,
                exclude_expired=req.exclude_expired,
                max_results=req.max_results,
            ))
            results[config_name] = single
        except Exception as e:
            results[config_name] = {"error": str(e)}
    return results
