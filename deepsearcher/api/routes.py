"""
API Route Definitions

This module defines the FastAPI routes and their handler functions for the Rbase API.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, List, Dict, Optional

import random
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from deepsearcher import configuration

from deepsearcher.db.async_mysql_connection import get_mysql_pool
from deepsearcher.agent.summary_rag import SummaryRag
from deepsearcher.agent.discuss_agent import DiscussAgent
from deepsearcher.rbase_db_loading import load_articles_by_channel, load_articles_by_article_ids
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
    get_discuss_history,
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
    DiscussAIReplyRequest,
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

        if request.depress_cache == DepressCache.DISABLE:
            ai_response = await get_response_by_request_hash(ai_request.request_hash)
        else:
            ai_response = None

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
    summary="Discussion Topic Creation API",
    description="""
    Create a new discussion topic or return existing topic UUID.
    
    - Supports author, topic, and paper related types
    - Uses request_hash and user_hash to check for existing topics
    - Returns topic UUID
    """,
)
async def api_create_discuss(request: DiscussCreateRequest):
    """
    Create a discussion topic or return existing topic UUID.

    Args:
        request (DiscussCreateRequest): Request parameters for creating discussion topic

    Returns:
        DiscussCreateResponse: Response containing topic UUID
    """
    try:
        discuss_thread = initialize_discuss_thread(request)
        result = await get_discuss_thread_by_request_hash(discuss_thread.request_hash, discuss_thread.user_hash)
        
        if result:
            # If exists, return UUID directly
            return DiscussCreateResponse(
                code=0,
                message="success",
                thread_uuid=result.uuid
            )
        else:
            # If not exists, create new thread
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
    summary="Discussion Content Posting API",
    description="""
    Post discussion content to specified topic.
    
    - Specify topic UUID and reply UUID
    - User hash and ID must be provided
    - Returns success or failure status
    """,
)
async def api_post_discuss(request: DiscussPostRequest):
    """
    Post discussion content to specified topic.

    Args:
        request (DiscussPostRequest): Request parameters for posting discussion content

    Returns:
        DiscussPostResponse: Response containing posting result
    """
    try:
        # Verify if discussion topic exists
        thread = await get_discuss_thread_by_uuid(request.thread_uuid)
        if not thread:
            return JSONResponse(
                status_code=400,
                content=ExceptionResponse(
                    code=400, 
                    message=f"讨论话题不存在: {request.thread_uuid}"
                ).model_dump()
            )
        
        # If reply UUID is specified, verify if reply object exists
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
        
        # Create and save discussion content
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

@router.post(
    "/generate/ai_reply",
    summary="AI Discussion Reply API",
    description="""
    Automatically generate AI reply based on user discussion content.
    
    - Supports streaming output
    - Generates appropriate reply based on discussion topic and history
    - Returns generated reply content UUID
    """,
)
async def api_ai_reply_discuss(request: DiscussAIReplyRequest):
    """
    Generate AI reply for discussion content.

    Args:
        request (DiscussAIReplyRequest): AI reply discussion request

    Returns:
        StreamingResponse: Streaming response with AI generated reply content
    """
    try:
        # 1. Get discussion topic data
        thread = await get_discuss_thread_by_uuid(request.thread_uuid)
        if not thread:
            return JSONResponse(
                status_code=400,
                content=ExceptionResponse(
                    code=400, 
                    message=f"讨论话题不存在: {request.thread_uuid}"
                ).model_dump()
            )
        
        # 2. Get discussion content to reply to
        reply_discuss = await get_discuss_by_uuid(request.reply_uuid)
        if not reply_discuss:
            return JSONResponse(
                status_code=400,
                content=ExceptionResponse(
                    code=400, 
                    message=f"回复对象不存在: {request.reply_uuid}"
                ).model_dump()
            )
        
        # Create AI reply discussion object
        ai_discuss = Discuss(
            uuid="",
            thread_id=thread.id,
            thread_uuid=thread.uuid,
            reply_id=reply_discuss.id if reply_discuss else None,
            reply_uuid=reply_discuss.uuid if reply_discuss else "",
            depth=reply_discuss.depth + 1 if reply_discuss else 0,
            content="",  # Content will be updated during streaming generation
            role=DiscussRole.ASSISTANT,
            tokens={},
            usage={},
            status=AIResponseStatus.GENERATING,
            user_id=request.user_id,
            created=time.time(),
            modified=time.time(),
        )
        ai_discuss.create_uuid()
        
        # Save empty content, get ID, update content later
        discuss_id = await save_discuss_to_db(ai_discuss)
        ai_discuss.id = discuss_id
        
        # 3. Return streaming response
        return StreamingResponse(
            generate_ai_reply_stream(ai_discuss, thread, reply_discuss),
            media_type="text/event-stream"
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
    
    if related_type == RelatedType.CHANNEL or related_type == RelatedType.COLUMN:
        articles = await load_articles_by_channel(
            ai_request.params.get("channel_id", 0),
            ai_request.params.get("term_tree_node_ids", []))
    elif related_type == RelatedType.ARTICLE:
        articles = await load_articles_by_article_ids(
            ai_request.params.get("article_ids", []))
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
            term_tree_node_ids=thread.params.get("term_ids", None),
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

async def generate_ai_reply_stream(ai_discuss: Discuss, thread: DiscussThread, reply_discuss: Discuss) -> AsyncGenerator[bytes, None]:
    """
    Generate streaming response for AI reply discussion content.

    Args:
        ai_discuss: AI reply discussion object
        thread: Discussion topic
        reply_discuss: Discussion content to reply to

    Yields:
        bytes: Streamed content chunks
    """
    try:
        # Get discussion background information
        background = await get_thread_background(thread)
        
        # Get history records (last 10)
        history = await get_discuss_history(thread.id, reply_discuss.id, limit=10)
        
        # Create DiscussAgent instance
        discuss_agent = DiscussAgent(
            llm=configuration.writing_llm,
            reasoning_llm=configuration.reasoning_llm,
            translator=configuration.academic_translator,
            embedding_model=configuration.embedding_model,
            vector_db=configuration.vector_db,
        )
        
        # Send role message
        role_chunk = {
            "id": f"chatcmpl-{ai_discuss.id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "rbase-discuss-agent",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(role_chunk)}\n\n".encode('utf-8')
        
        # Update AI discussion status
        ai_discuss.is_generating = 1
        ai_discuss.tokens["generating"] = []
        await save_discuss_to_db(ai_discuss)
        
        # Possible user action
        user_action = "浏览学术内容"
        
        # Extract user query content
        query = reply_discuss.content
        
        # Set request parameters (can be adjusted based on actual needs)
        request_params = {}
        
        # Call DiscussAgent to generate reply
        for chunk in discuss_agent.query_generator(
            query=query,
            user_action=user_action,
            background=background,
            history=history,
            request_params=request_params,
        ):
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                stop = chunk.choices[0].finish_reason == "stop"
                
                if hasattr(delta, "content") and delta.content is not None:
                    # Update content
                    ai_discuss.content += delta.content
                    ai_discuss.tokens["generating"].append(delta.content)
                    await save_discuss_to_db(ai_discuss)
                    
                    # Build response chunk
                    content_chunk = {
                        "id": f"chatcmpl-{ai_discuss.id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "rbase-discuss-agent",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta.content},
                            "finish_reason": None if not stop else "stop"
                        }]
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n".encode('utf-8')
        
        # Update final status
        ai_discuss.is_generating = 0
        ai_discuss.tokens["generating"] = []
        ai_discuss.status = AIResponseStatus.FINISHED
        if hasattr(discuss_agent, "usage"):
            ai_discuss.usage = discuss_agent.usage
        await save_discuss_to_db(ai_discuss)
        
        # Send completion marker
        yield "data: [DONE]\n\n".encode('utf-8')
        
    except Exception as e:
        # Error handling
        ai_discuss.content += f"\n\n生成回复时发生错误: {str(e)}"
        ai_discuss.status = AIResponseStatus.ERROR
        await save_discuss_to_db(ai_discuss)
        
        error_chunk = {
            "id": f"chatcmpl-{ai_discuss.id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "rbase-discuss-agent",
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n生成回复时发生错误: {str(e)}"},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')
        yield "data: [DONE]\n\n".encode('utf-8')

async def get_thread_background(thread: DiscussThread) -> str:
    """
    Get background information for discussion thread.
    
    Get corresponding background information based on thread related type (channel, column, article).
    
    Args:
        thread: Discussion thread object
        
    Returns:
        str: Background information text
    """
    # If preset background exists, return directly
    if hasattr(thread, "background") and thread.background:
        return thread.background
    
    # Get background information based on related type
    try:
        if thread.related_type == RelatedType.CHANNEL or thread.related_type == RelatedType.COLUMN:
            channel_id = thread.params.get("channel_id", 0)
            if channel_id:
                # Try to get channel information from cached AI summary
                summary_request = SummaryRequest(
                    related_type=RelatedType.CHANNEL,
                    related_id=channel_id,
                    term_tree_node_ids=thread.params.get("term_tree_node_ids", []),
                    depress_cache=DepressCache.DISABLE,
                    stream=False
                )
                ai_request = initialize_ai_request_by_summary(summary_request)
                ai_response = await get_response_by_request_hash(ai_request.request_hash)
                if ai_response and ai_response.content:
                    return ai_response.content
                
                # Or get channel basic information directly from database
                rbase_config = configuration.config.rbase_settings
                pool = await get_mysql_pool(rbase_config.get("database"))
                async with pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        sql = "SELECT id, name FROM base WHERE id = %s"
                        await cursor.execute(sql, (channel_id,))
                        result = await cursor.fetchone()
                        if result:
                            return f"频道: {result['name']}"
        elif thread.related_type == RelatedType.ARTICLE:
            article_id = thread.params.get("article_id", 0)
            if article_id:
                # Get article basic information from database
                rbase_config = configuration.config.rbase_settings
                pool = await get_mysql_pool(rbase_config.get("database"))
                async with pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        sql = """
                        SELECT title, abstract, journal_name, authors, pubdate
                        FROM article WHERE id = %s
                        """
                        await cursor.execute(sql, (article_id,))
                        result = await cursor.fetchone()
                        if result:
                            authors = result["authors"].split(",")
                            if len(authors) > 3:
                                authors = authors[:3] + ["et al."]
                            pub_year = result["pubdate"].year if hasattr(result["pubdate"], "year") else "未知"
                            
                            return (
                                f"文章标题: {result['title']}\n\n"
                                f"作者: {', '.join(authors)}\n\n"
                                f"期刊: {result['journal_name']} ({pub_year})\n\n"
                                f"摘要: {result['abstract']}"
                            )
    except Exception as e:
        raise Exception(f"获取讨论背景信息失败: {e}")
    
    # Return empty background by default
    return ""
