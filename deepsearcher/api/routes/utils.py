"""
Shared Utilities

This module contains shared utility functions for API routes.
"""

import json
from deepsearcher import configuration
from deepsearcher.api.models import RelatedType, SummaryRequest
from deepsearcher.api.rbase_util import (
    save_request_to_db,
    save_response_to_db,
    update_ai_content_to_discuss,
)
from deepsearcher.agent.summary_rag import SummaryRag
from deepsearcher.rbase_db_loading import load_articles_by_channel, load_articles_by_article_ids
from deepsearcher.rbase.ai_models import (
    AIContentRequest,
    AIRequestStatus,
    AIResponseStatus,
    initialize_ai_content_response,
)

async def generate_ai_content(ai_request: AIContentRequest, related_type: RelatedType, summary_request: SummaryRequest, purpose: str = "") -> str:
    """
    Create AI content based on the request and related type.

    Args:
        ai_request (AIContentRequest): The AI content request
        related_type (RelatedType): The type of related content
        summary_request (SummaryRequest): Optional summary request for discuss integration

    Returns:
        str: The generated content
    """
    request_id = await save_request_to_db(ai_request)
    ai_request.id = request_id

    ai_response = initialize_ai_content_response(ai_request, ai_request.id)
    response_id = await save_response_to_db(ai_response)
    ai_response.id = response_id

    if related_type == RelatedType.CHANNEL:
        articles = await load_articles_by_channel(
            ai_request.params.get("channel_id", 0), 
            ai_request.params.get("term_tree_node_ids", []))
    elif related_type == RelatedType.COLUMN:
        articles = await load_articles_by_channel(
            ai_request.params.get("channel_id", 0),
            ai_request.params.get("term_tree_node_ids", []))
    elif related_type == RelatedType.ARTICLE:
        articles = await load_articles_by_article_ids(
            [ai_request.params.get("article_id")])
    else:
        articles = []

    summary_rag = SummaryRag(
        reasoning_llm=configuration.reasoning_llm,
        writing_llm=configuration.writing_llm,
    )

    params = {"min_words": 500, "max_words": 800, "question_count": ai_request.params.get("question_count", 3)}
    summary, _, usage = summary_rag.query(
        query=ai_request.query,
        articles=articles,
        params=params,
        purpose=purpose,
        verbose=False,
    )

    ai_request.status = AIRequestStatus.FINISHED
    await save_request_to_db(ai_request)

    ai_response.content = summary
    ai_response.tokens = json.dumps({"generating": []})
    ai_response.usage = json.dumps(usage.to_dict())
    ai_response.status = AIResponseStatus.FINISHED
    await save_response_to_db(ai_response)

    if summary_request:
        await update_ai_content_to_discuss(ai_response, summary_request.discuss_thread_uuid, summary_request.discuss_reply_uuid)

    return summary 