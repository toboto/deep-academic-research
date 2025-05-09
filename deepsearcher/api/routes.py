"""
API Route Definitions

This module defines the FastAPI routes and their handler functions for the Rbase API.
"""

import asyncio
import json
import time
from typing import AsyncGenerator

import random
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError
from deepsearcher import configuration

from deepsearcher.agent.summary_rag import SummaryRag
from deepsearcher.rbase_db_loading import load_articles_by_channel, load_articles_by_column
from deepsearcher.api.rbase_util import get_response_by_request_hash, save_request_to_db, save_response_to_db
from deepsearcher.api.models import (
    RelatedType,
    SummaryRequest,
    SummaryResponse,
    QuestionRequest,
    QuestionResponse,
    AIContentRequest,
    AIRequestStatus,
    AIContentResponse,
    AIResponseStatus,
    initialize_ai_request_by_summary,
    initialize_ai_request_by_question,
    initialize_ai_content_response,
    ExceptionResponse,
)

router = APIRouter()

@router.post(
    "/generate/summary",
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
        ai_request = initialize_ai_request_by_summary(request)

        ai_response = get_response_by_request_hash(ai_request.request_hash)

        if request.stream:
            if not ai_response:
                return create_summary_stream(ai_request, request.related_type)
            else:
                return StreamingResponse(generate_text_stream(ai_response.content, ai_response.id), 
                                         media_type="text/event-stream")
        else:
            resp = SummaryResponse(code=0, message="success")
            if not ai_response:
                summary = generate_ai_content(ai_request, request.related_type)
                resp.setContent(summary)
            else:
                resp.setContent(ai_response.content)

            return resp
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        )

@router.post(
    "/generate/questions",
    summary="AI Question Recommendation API",
    description="""
    Generate AI recommended questions.
    
    - Supports author, topic, and paper related types
    - Optional cache usage
    - Supports streaming response
    """,
)
async def api_generate_questions(request: QuestionRequest):
    """
    Generate AI recommended questions based on the request.

    Args:
        request (QuestionRequest): The question request parameters

    Returns:
        QuestionResponse: The response containing recommended questions

    Raises:
        HTTPException: When request parameters are invalid or processing fails
    """
    try:
        ai_request = initialize_ai_request_by_question(request)
        ai_response = get_response_by_request_hash(ai_request.request_hash)
        question_response = QuestionResponse(code=0, message="success")
        if not ai_response:
            summary = generate_ai_content(ai_request, request.related_type) 
            question_response.setQuestions(summary)
        else:
            question_response.setQuestions(ai_response.content)
        return question_response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        )


def create_summary_stream(ai_request: AIContentRequest, related_type: RelatedType) -> StreamingResponse:
    """
    Create a streaming response for summary generation.

    Args:
        ai_request (AIContentRequest): The AI content request
        related_type (RelatedType): The type of related content

    Returns:
        StreamingResponse: The streaming response object
    """
    rbase_config = configuration.config.rbase_settings
    request_id = save_request_to_db(ai_request)
    ai_request.id = request_id

    ai_response = initialize_ai_content_response(ai_request, ai_request.id)
    response_id = save_response_to_db(ai_response)
    ai_response.id = response_id

    return StreamingResponse(generate_summary_stream(rbase_config, ai_request, ai_response, related_type), 
                             media_type="text/event-stream")

    
async def generate_summary_stream(rbase_config: dict, ai_request: AIContentRequest, ai_response: AIContentResponse, related_type: RelatedType) -> AsyncGenerator[bytes, None]:
    """
    Generate a streaming response for summary content.

    Args:
        rbase_config (dict): The Rbase configuration
        ai_request (AIContentRequest): The AI content request
        ai_response (AIContentResponse): The AI content response
        related_type (RelatedType): The type of related content

    Yields:
        bytes: Chunks of the streaming response data
    """
    if related_type == RelatedType.CHANNEL:
        articles = load_articles_by_channel(rbase_config, ai_request.params.get("channel_id", 0))
    elif related_type == RelatedType.COLUMN:
        articles = load_articles_by_column(rbase_config, ai_request.params.get("column_id", 0))
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
    save_request_to_db(ai_request)

    params = {"min_words": 500, "max_words": 800, "question_count": 3}
    for chunk in summary_rag.query_generator(query=ai_request.query, articles=articles, params=params):
        if hasattr(chunk, "usage") and chunk.usage:
            ai_response.usage = chunk.usage.to_dict()
            save_response_to_db(ai_response)

        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            stop = chunk.choices[0].finish_reason == "stop"
            if hasattr(delta, "content") and delta.content is not None:
                ai_response.is_generating = 1
                ai_response.content += delta.content
                ai_response.tokens["generating"].append(delta.content)
                save_response_to_db(ai_response)
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
    save_request_to_db(ai_request)

    ai_response.is_generating = 0
    ai_response.tokens["generating"] = []
    ai_response.status = AIResponseStatus.FINISHED
    save_response_to_db(ai_response)
    
    yield "data: [DONE]\n\n".encode('utf-8')


async def generate_text_stream(text: str, response_id: int) -> AsyncGenerator[bytes, None]:
    """
    Generate a streaming response for text content.

    Args:
        text (str): The text content to be streamed
        response_id (int): The unique identifier for this response

    Yields:
        bytes: Chunks of the streaming response data
    """
    # Send role message
    role_chunk = {
        "id": f"chatcmpl-{response_id}",
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
    
    # 将文本均匀划分为30段
    total_chunks = 30
    text_length = len(text)
    chunk_size = max(1, text_length // total_chunks)
    chunks = []
    
    for i in range(total_chunks - 1):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        if start_idx < text_length:
            chunks.append(text[start_idx:end_idx])
    
    # 添加最后一段，包含所有剩余文本
    if (total_chunks - 1) * chunk_size < text_length:
        chunks.append(text[(total_chunks - 1) * chunk_size:])
    
    # 移除空块
    chunks = [chunk for chunk in chunks if chunk]
    
    # 发送每一段内容
    for i, chunk in enumerate(chunks):
        await asyncio.sleep(random.uniform(0.1, 0.4))  # random delay
        content_chunk = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "rbase-summary-rag",
            "choices": [{
                "index": i,
                "delta": {"content": chunk},
                "finish_reason": None if i < len(chunks) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(content_chunk)}\n\n".encode('utf-8')
    
    # Send end marker
    yield "data: [DONE]\n\n".encode('utf-8')


def generate_ai_content(ai_request: AIContentRequest, related_type: RelatedType) -> str:
    """
    Create AI content based on the request and related type.

    Args:
        ai_request (AIContentRequest): The AI content request
        related_type (RelatedType): The type of related content

    Returns:
        str: The generated content
    """
    rbase_config = configuration.config.rbase_settings
    request_id = save_request_to_db(ai_request)
    ai_request.id = request_id

    ai_response = initialize_ai_content_response(ai_request, ai_request.id)
    response_id = save_response_to_db(ai_response)
    ai_response.id = response_id

    if related_type == RelatedType.CHANNEL:
        articles = load_articles_by_channel(rbase_config, ai_request.params.get("channel_id", 0))
    elif related_type == RelatedType.COLUMN:
        articles = load_articles_by_column(rbase_config, ai_request.params.get("column_id", 0))
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
        verbose=False,
    )

    ai_request.status = AIRequestStatus.FINISHED
    save_request_to_db(ai_request)

    ai_response.content = summary
    ai_response.tokens = json.dumps({"generating": []})
    ai_response.usage = json.dumps(usage.to_dict())
    ai_response.status = AIResponseStatus.FINISHED
    save_response_to_db(ai_response)

    return summary