"""
API路由定义

本模块定义了FastAPI接口的路由和处理函数。
"""

import asyncio
import json
import time
import os
from typing import AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
    AIContentType,
    AIContentRequest,
    AIRequestStatus,
    AIContentResponse,
    AIResponseStatus,
    initialize_ai_request_by_summary,
    initialize_ai_request_by_question,
    initialize_ai_content_response,
)

router = APIRouter()


async def generate_text_stream(text: str, response_id: int) -> AsyncGenerator[bytes, None]:
    """
    生成流式响应

    Args:
        text: 要发送的文本内容

    Yields:
        bytes: 流式响应数据
    """
    # 发送role消息
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
    
    # 发送内容消息
    words = text.split()
    for i, word in enumerate(words):
        await asyncio.sleep(0.5)  # 模拟延迟
        content_chunk = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "rbase-summary-rag",
            "choices": [{
                "index": 0,
                "delta": {"content": word},
                "finish_reason": None if i < len(words) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(content_chunk)}\n\n".encode('utf-8')
    
    # 发送结束标记
    yield "data: [DONE]\n\n".encode('utf-8')


@router.post(
    "/generate/summary",
    response_model=SummaryResponse,
    summary="AI概述接口",
    description="""
    生成AI概述内容。
    
    - 支持作者、主题、论文三种关联类型
    - 可选择是否使用缓存
    - 支持流式响应
    """,
)
async def api_generate_summary(request: SummaryRequest):
    """
    生成AI概述内容

    Args:
        request: 请求参数

    Returns:
        SummaryResponse或StreamingResponse: 响应结果

    Raises:
        HTTPException: 当请求参数无效或处理失败时抛出
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
                summary = create_ai_content(ai_request, request.related_type)
                resp.setContent(summary)
            else:
                resp.setContent(ai_response.content)

            return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generate/questions",
    response_model=QuestionResponse,
    summary="AI推荐问题接口",
    description="""
    生成AI推荐问题。
    
    - 支持作者、主题、论文三种关联类型
    - 可选择是否使用缓存
    - 支持流式响应
    """,
)
async def api_generate_questions(request: QuestionRequest) -> QuestionResponse:
    """
    生成AI推荐问题

    Args:
        request: 请求参数

    Returns:
        QuestionResponse: 响应结果

    Raises:
        HTTPException: 当请求参数无效或处理失败时抛出
    """
    try:
        ai_request = initialize_ai_request_by_question(request)
        ai_response = get_response_by_request_hash(ai_request.request_hash)
        question_response = QuestionResponse(code=0, message="success")
        if not ai_response:
            summary = create_ai_content(ai_request, request.related_type) 
            question_response.setQuestions(summary)
        else:
            question_response.setQuestions(ai_response.content)
        return question_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_summary_stream(ai_request: AIContentRequest, related_type: RelatedType) -> StreamingResponse:
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
    生成流式响应

    Args:
        text: 要发送的文本内容

    Yields:
        bytes: 流式响应数据
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
    # 发送role消息

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


def create_ai_content(ai_request: AIContentRequest, related_type: RelatedType) -> str:
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