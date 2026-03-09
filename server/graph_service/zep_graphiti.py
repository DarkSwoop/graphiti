import json
import logging
import types
from typing import Annotated

import openai as _openai
from fastapi import Depends, HTTPException
from graphiti_core import Graphiti  # type: ignore
from graphiti_core.edges import EntityEdge  # type: ignore
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.llm_client import LLMClient  # type: ignore
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore
from pydantic import BaseModel

from graph_service.config import ZepEnvDep
from graph_service.dto import FactResult

logger = logging.getLogger(__name__)


async def _generate_response_lmstudio(
    self,
    messages,
    response_model: type[BaseModel] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model_size: ModelSize = ModelSize.medium,
):
    """LM Studio compatible: uses chat.completions API with json_schema format.

    The upstream OpenAIClient._create_structured_completion uses the OpenAI
    Responses API (client.responses.parse) which LM Studio does not implement.
    This replacement uses chat.completions with response_format type=json_schema.
    """
    openai_messages = self._convert_messages_to_openai_format(messages)
    model = self._get_model_for_size(model_size)

    try:
        request_kwargs: dict = {
            'model': model,
            'messages': openai_messages,
            'temperature': self.temperature,
            'max_tokens': max_tokens or self.max_tokens,
        }

        if response_model:
            schema = response_model.model_json_schema()
            request_kwargs['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': response_model.__name__,
                    'schema': schema,
                },
            }
        else:
            request_kwargs['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'json_response',
                    'schema': {'type': 'object'},
                },
            }

        response = await self.client.chat.completions.create(**request_kwargs)

        result = response.choices[0].message.content or '{}'
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

        result_dict = json.loads(result)

        # Validate against Pydantic model (raises ValidationError → triggers retry)
        if response_model:
            response_model.model_validate(result_dict)

        return result_dict, input_tokens, output_tokens

    except _openai.LengthFinishReasonError as e:
        raise Exception(f'Output length exceeded max tokens {self.max_tokens}: {e}') from e
    except _openai.RateLimitError:
        raise RateLimitError
    except _openai.AuthenticationError:
        raise
    except Exception as e:
        error_msg = str(e)
        if 'connection' in error_msg.lower():
            logger.error(f'Connection error communicating with LLM API: {e}')
        else:
            logger.error(f'Error in generating LLM response: {e}')
        raise


class ZepGraphiti(Graphiti):
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
        super().__init__(uri, user, password, llm_client)

    async def save_entity_node(self, name: str, uuid: str, group_id: str, summary: str = ''):
        new_node = EntityNode(
            name=name,
            uuid=uuid,
            group_id=group_id,
            summary=summary,
        )
        await new_node.generate_name_embedding(self.embedder)
        await new_node.save(self.driver)
        return new_node

    async def get_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            return edge
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_group(self, group_id: str):
        try:
            edges = await EntityEdge.get_by_group_ids(self.driver, [group_id])
        except GroupsEdgesNotFoundError:
            logger.warning(f'No edges found for group {group_id}')
            edges = []

        nodes = await EntityNode.get_by_group_ids(self.driver, [group_id])

        episodes = await EpisodicNode.get_by_group_ids(self.driver, [group_id])

        for edge in edges:
            await edge.delete(self.driver)

        for node in nodes:
            await node.delete(self.driver)

        for episode in episodes:
            await episode.delete(self.driver)

    async def delete_entity_edge(self, uuid: str):
        try:
            edge = await EntityEdge.get_by_uuid(self.driver, uuid)
            await edge.delete(self.driver)
        except EdgeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e

    async def delete_episodic_node(self, uuid: str):
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
            await episode.delete(self.driver)
        except NodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message) from e


def _configure_client(client: ZepGraphiti, settings) -> None:
    """Apply LLM + embedder config from settings to a ZepGraphiti client."""
    from openai import AsyncOpenAI

    if settings.openai_base_url is not None:
        client.llm_client.config.base_url = settings.openai_base_url
        client.embedder.config.base_url = settings.openai_base_url
    if settings.openai_api_key is not None:
        client.llm_client.config.api_key = settings.openai_api_key
        client.embedder.config.api_key = settings.openai_api_key
    if settings.model_name is not None:
        client.llm_client.model = settings.model_name
        # Also override small_model (defaults to gpt-4.1-nano which doesn't exist in LM Studio)
        client.llm_client.small_model = settings.model_name
    if settings.embedding_model_name is not None:
        client.embedder.config.embedding_model = settings.embedding_model_name

    # CRITICAL: Re-create the underlying AsyncOpenAI HTTP clients.
    # The originals were created in __init__ with default config (api.openai.com).
    # Changing config attributes after init does NOT update the HTTP client.
    client.llm_client.client = AsyncOpenAI(
        api_key=client.llm_client.config.api_key,
        base_url=client.llm_client.config.base_url,
    )
    client.embedder.client = AsyncOpenAI(
        api_key=client.embedder.config.api_key,
        base_url=client.embedder.config.base_url,
    )

    # Monkey-patch _generate_response to use Chat Completions API instead of
    # OpenAI Responses API (responses.parse), which LM Studio doesn't support.
    client.llm_client._generate_response = types.MethodType(
        _generate_response_lmstudio, client.llm_client
    )


async def get_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    _configure_client(client, settings)

    try:
        yield client
    finally:
        await client.close()


async def create_configured_graphiti() -> ZepGraphiti:
    """Create an app-scoped Graphiti client (not tied to a request lifecycle)."""
    from graph_service.config import get_settings
    settings = get_settings()
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    _configure_client(client, settings)
    return client


async def initialize_graphiti(settings: ZepEnvDep):
    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await client.build_indices_and_constraints()


def get_fact_result_from_edge(edge: EntityEdge):
    return FactResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        created_at=edge.created_at,
        expired_at=edge.expired_at,
    )


ZepGraphitiDep = Annotated[ZepGraphiti, Depends(get_graphiti)]
