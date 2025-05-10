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
from deepsearcher import configuration

from deepsearcher.agent.summary_rag import SummaryRag
from deepsearcher.rbase_db_loading import load_articles_by_channel, load_articles_by_column
from deepsearcher.api.rbase_util import (
    get_response_by_request_hash, 
    save_request_to_db, 
    save_response_to_db,
    get_discuss_thread_by_request_hash,
    get_discuss_thread_by_id,
    save_discuss_thread_to_db,
    get_discuss_thread_by_uuid,
    save_discuss_to_db,
    get_discuss_by_uuid,
)
from deepsearcher.api.models import (
    RelatedType,
    SummaryRequest,
    SummaryResponse,
    QuestionRequest,
    QuestionResponse,
    ExceptionResponse,
    DepressCache,
    DiscussCreateRequest,
    DiscussCreateResponse,
    DiscussPostRequest,
    DiscussPostResponse,
)

from deepsearcher.rbase.ai_models import (
    AIContentRequest,
    AIRequestStatus,
    AIResponseStatus,
    Discuss, 
    DiscussThread,
    DiscussRole,
    initialize_ai_request_by_summary,
    initialize_ai_request_by_question,
    initialize_ai_content_response,
    initialize_discuss_thread,
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

        ai_response = await get_response_by_request_hash(ai_request.request_hash)

        if request.stream:
            if not ai_response:
                return create_summary_stream(ai_request, request.related_type)
            else:
                return StreamingResponse(generate_text_stream(ai_response.content, ai_response.id), 
                                         media_type="text/event-stream")
        else:
            resp = SummaryResponse(code=0, message="success")
            if not ai_response:
                summary = await generate_ai_content(ai_request, request.related_type)
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
        ai_response = await get_response_by_request_hash(ai_request.request_hash)
        question_response = QuestionResponse(code=0, message="success")
        if not ai_response:
            summary = await generate_ai_content(ai_request, request.related_type) 
            question_response.setQuestions(summary)
        else:
            question_response.setQuestions(ai_response.content)
        return question_response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        )

@router.post(
    "/generate/discuss_create",
    summary="创建讨论话题接口",
    description="""
    创建新的讨论话题或返回已存在的讨论话题UUID。
    
    - 支持作者、主题和论文相关类型
    - 使用request_hash和user_hash检查是否存在相同话题
    - 返回话题UUID
    """,
)
async def api_create_discuss(request: DiscussCreateRequest):
    """
    创建讨论话题或返回已存在的讨论话题UUID。

    Args:
        request (DiscussCreateRequest): 创建讨论话题的请求参数

    Returns:
        DiscussCreateResponse: 包含话题UUID的响应
    """
    try:
        discuss_thread = initialize_discuss_thread(request)
        result = await get_discuss_thread_by_request_hash(discuss_thread.request_hash, discuss_thread.user_hash)
        
        if result:
            # 如果存在，直接返回uuid
            return DiscussCreateResponse(
                code=0,
                message="success",
                thread_uuid=result.uuid
            )
        else:
            # 如果不存在，创建新线程
            await save_discuss_thread_to_db(discuss_thread)
            return DiscussCreateResponse(
                code=0,
                message="success",
                thread_uuid=discuss_thread.uuid
            )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        )

@router.post(
    "/generate/discuss_post",
    summary="发布讨论内容接口",
    description="""
    发布讨论内容到指定话题。
    
    - 指定话题UUID和回复UUID
    - 发布内容必须提供用户hash和ID
    - 返回成功或失败状态
    """,
)
async def api_post_discuss(request: DiscussPostRequest):
    """
    发布讨论内容到指定话题。

    Args:
        request (DiscussPostRequest): 发布讨论内容的请求参数

    Returns:
        DiscussPostResponse: 发布结果的响应
    """
    try:
        # 验证讨论话题是否存在
        thread = await get_discuss_thread_by_uuid(request.thread_uuid)
        if not thread:
            return JSONResponse(
                status_code=400,
                content=ExceptionResponse(
                    code=400, 
                    message=f"讨论话题不存在: {request.thread_uuid}"
                ).model_dump()
            )
        
        # 如果指定了回复UUID，验证回复对象是否存在
        if request.reply_uuid and request.reply_uuid != "":
            reply_discuss = await get_discuss_by_uuid(request.reply_uuid)
            if not reply_discuss:
                return JSONResponse(
                    status_code=400,
                    content=ExceptionResponse(
                        code=400, 
                        message=f"回复对象不存在: {request.reply_uuid}"
                    ).model_dump()
                )
        else:
            reply_discuss = None
        
        # 创建并保存讨论内容
        discuss = initialize_discuss_by_post_request(request, thread, reply_discuss)
        content_id = await save_discuss_to_db(discuss)
        if content_id:
            return DiscussPostResponse(
                code=0,
                message="success",
                uuid=discuss.uuid,
                depth=discuss.depth
            )
        else:
            return JSONResponse(
                status_code=500,
                content=ExceptionResponse(code=500, message="保存讨论内容失败").model_dump()
            )
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
    
    return StreamingResponse(generate_summary_stream(rbase_config, ai_request, related_type), 
                             media_type="text/event-stream")

    
async def generate_summary_stream(rbase_config: dict, ai_request: AIContentRequest, related_type: RelatedType) -> AsyncGenerator[bytes, None]:
    """
    Generate a streaming response for summary content.

    Args:
        rbase_config (dict): The Rbase configuration
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
    await save_request_to_db(ai_request)

    params = {"min_words": 500, "max_words": 800, "question_count": 3}
    for chunk in summary_rag.query_generator(query=ai_request.query, articles=articles, params=params):
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


async def generate_ai_content(ai_request: AIContentRequest, related_type: RelatedType) -> str:
    """
    Create AI content based on the request and related type.

    Args:
        ai_request (AIContentRequest): The AI content request
        related_type (RelatedType): The type of related content

    Returns:
        str: The generated content
    """
    rbase_config = configuration.config.rbase_settings
    request_id = await save_request_to_db(ai_request)
    ai_request.id = request_id

    ai_response = initialize_ai_content_response(ai_request, ai_request.id)
    response_id = await save_response_to_db(ai_response)
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
    await save_request_to_db(ai_request)

    ai_response.content = summary
    ai_response.tokens = json.dumps({"generating": []})
    ai_response.usage = json.dumps(usage.to_dict())
    ai_response.status = AIResponseStatus.FINISHED
    await save_response_to_db(ai_response)

    return summary

async def get_discuss_background(thread_id: int) -> str:
    """
    Get the background of a discuss thread.
    """
    thread = await get_discuss_thread_by_id(thread_id)
    if not thread:
        return "" 

    if thread.background and len(thread.background) > 0:
        return thread.background
    
    if RelatedType.IsValid(thread.related_type):
        if thread.related_type == RelatedType.CHANNEL:
            related_id = thread.params.get("channel_id", 0) 
        elif thread.related_type == RelatedType.COLUMN:
            related_id = thread.params.get("column_id", 0)
        elif thread.related_type == RelatedType.ARTICLE:
            related_id = thread.params.get("article_id", 0)
        else:
            related_id = 0
 
        ai_request = initialize_ai_request_by_summary(SummaryRequest(
            related_type=thread.related_type,
            related_id=related_id,
            term_ids=thread.params.get("term_ids", None),
            ver=thread.params.get("ver", 0),
            depress_cache=DepressCache.DISABLE,
            stream=False,
        ))
        ai_response = await get_response_by_request_hash(ai_request.hash)
        if ai_response:
            return ai_response.content
        else:
            return ""
    else:
        return ""

def initialize_discuss_by_post_request(request: DiscussPostRequest, thread: DiscussThread, replyDiscuss: Discuss) -> Discuss:
    """
    Initialize a discuss object based on the post request.
    """
    discuss = Discuss(
        uuid="",
        thread_id=thread.id,
        thread_uuid=thread.uuid,
        reply_id=replyDiscuss.id if replyDiscuss else 0,
        reply_uuid=replyDiscuss.uuid if replyDiscuss else "",
        depth=replyDiscuss.depth + 1 if replyDiscuss else 0,
        content=request.content,
        role=DiscussRole.USER,
        tokens={},
        usage={},
        status=AIResponseStatus.FINISHED,
        user_id=request.user_id,
        created=time.time(),
        modified=time.time(),
    )
    discuss.create_uuid()
    return discuss
