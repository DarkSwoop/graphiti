from datetime import datetime, timezone

from fastapi import APIRouter, status

from graphiti_core.search.search_config import (
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
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

from graph_service.dto import (
    AdvancedSearchQuery,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.dto.retrieve import AdvancedSearchResults
from graph_service.zep_graphiti import ZepGraphitiDep, build_advanced_results, get_fact_result_from_edge

router = APIRouter()


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.post('/search-advanced', status_code=status.HTTP_200_OK)
async def search_advanced(
    query: AdvancedSearchQuery, graphiti: ZepGraphitiDep
) -> AdvancedSearchResults:
    search_filter = SearchFilters()
    if query.exclude_expired:
        search_filter = SearchFilters(
            expired_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]]
        )

    config = SearchConfig(
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
        limit=query.max_results,
    )

    search_results = await graphiti.search_(
        query=query.query,
        config=config,
        group_ids=query.group_ids,
        search_filter=search_filter,
    )

    await graphiti.resolve_missing_node_names(search_results)

    return build_advanced_results(search_results)


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
