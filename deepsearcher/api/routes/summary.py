"""
Summary Generation Routes

This module contains routes and functions for generating AI summaries.
"""

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from deepsearcher import configuration
from deepsearcher.api.models import (
    RelatedType,
    SummaryRequest,
    SummaryResponse,
    ExceptionResponse,
    DepressCache,
)
from deepsearcher.api.rbase_util import (
    get_response_by_request_hash,
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
    initialize_ai_request_by_summary,
    initialize_ai_content_response,
)
from .metadata import build_metadata
from .stream import generate_text_stream
from .utils import generate_ai_content

router = APIRouter()

@router.post(
    "/summary",
    summary="AI Summary Generation API",
    description="""
    Generate AI summary content.
    
    - Supports author, topic, and paper related types
    - Optional cache usage
    - Supports streaming response
    """,
)
async def api_generate_summary(request: SummaryRequest):
    """
    Generate AI summary content based on the request.

    Args:
        request (SummaryRequest): The summary request parameters

    Returns:
        Union[SummaryResponse, StreamingResponse]: The response result

    Raises:
        HTTPException: When request parameters are invalid or processing fails
    """
    try:
        metadata = await build_metadata(request.related_type, request.related_id, request.term_tree_node_ids)
        ai_request = initialize_ai_request_by_summary(request, metadata)

        if request.depress_cache == DepressCache.DISABLE:
            ai_response = await get_response_by_request_hash(ai_request.request_hash)
        else:
            ai_response = None

        if request.stream:
            if not ai_response:
                return create_summary_stream(ai_request, request.related_type, request)
            else:
                await update_ai_content_to_discuss(ai_response, request.discuss_thread_uuid, request.discuss_reply_uuid)
                return StreamingResponse(generate_text_stream(ai_response.content, ai_response.id), 
                                         media_type="text/event-stream")
        else:
            resp = SummaryResponse(code=0, message="success")
            if not ai_response:
                summary = await generate_ai_content(ai_request, request.related_type, request, "summary")
                resp.setContent(summary)
            else:
                await update_ai_content_to_discuss(ai_response, request.discuss_thread_uuid, request.discuss_reply_uuid)
                resp.setContent(ai_response.content)

            return resp
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        )

def create_summary_stream(ai_request: AIContentRequest, related_type: RelatedType, summary_request: SummaryRequest) -> StreamingResponse:
    """
    Create a streaming response for summary generation.

    Args:
        ai_request (AIContentRequest): The AI content request
        related_type (RelatedType): The type of related content

    Returns:
        StreamingResponse: The streaming response object
    """
    return StreamingResponse(generate_summary_stream(ai_request, related_type, summary_request), 
                             media_type="text/event-stream")

async def generate_summary_stream(ai_request: AIContentRequest, related_type: RelatedType, summary_request: SummaryRequest) -> AsyncGenerator[bytes, None]:
    """
    Generate a streaming response for summary content.

    Args:
        ai_request (AIContentRequest): The AI content request
        related_type (RelatedType): The type of related content

    Yields:
        bytes: Chunks of the streaming response data
    """
    # 保存请求到数据库
    request_id = await save_request_to_db(ai_request)
    ai_request.id = request_id

    # 初始化响应
    ai_response = initialize_ai_content_response(ai_request, ai_request.id)
    response_id = await save_response_to_db(ai_response)
    ai_response.id = response_id
    
    if related_type == RelatedType.CHANNEL or related_type == RelatedType.COLUMN:
        articles = await load_articles_by_channel(
            ai_request.params.get("channel_id", 0),
            ai_request.params.get("term_tree_node_ids", []),
            0, configuration.config.rbase_settings.get("api", {}).get("summary_article_reference_cnt", 10))
    elif related_type == RelatedType.ARTICLE:
        articles = await load_articles_by_article_ids(
            [ai_request.params.get("article_id")])
    else:
        articles = []

    summary_rag = SummaryRag(
        reasoning_llm=configuration.reasoning_llm,
        writing_llm=configuration.writing_llm,
    )

    # Send role message
    role_chunk = {
        "id": f"chatcmpl-{ai_response.id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "rbase-summary-rag",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n".encode('utf-8')

    ai_request.status = AIRequestStatus.HANDLING_REQ
    await save_request_to_db(ai_request)

    params = {"min_words": 500, "max_words": 800, "question_count": ai_request.params.get("question_count", 3)}
    for chunk in summary_rag.query_generator(query=ai_request.query, articles=articles, params=params, purpose=summary_request.purpose.value):
        if hasattr(chunk, "usage") and chunk.usage:
            ai_response.usage = chunk.usage.to_dict()
            await save_response_to_db(ai_response)

        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            stop = chunk.choices[0].finish_reason == "stop"
            if hasattr(delta, "content") and delta.content is not None:
                ai_response.is_generating = 1
                ai_response.content += delta.content
                ai_response.tokens["generating"].append(delta.content)
                await save_response_to_db(ai_response)
                content_chunk = {
                    "id": f"chatcmpl-{ai_response.id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "rbase-summary-rag",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta.content},
                        "finish_reason": None if not stop else "stop"
                    }]
                }
                yield f"data: {json.dumps(content_chunk)}\n\n".encode('utf-8')

    ai_request.status = AIRequestStatus.FINISHED
    await save_request_to_db(ai_request)

    ai_response.is_generating = 0
    ai_response.tokens["generating"] = []
    ai_response.status = AIResponseStatus.FINISHED
    await save_response_to_db(ai_response)

    await update_ai_content_to_discuss(ai_response, summary_request.discuss_thread_uuid, summary_request.discuss_reply_uuid)

    yield "data: [DONE]\n\n".encode('utf-8') 