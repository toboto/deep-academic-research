"""
Discussion Routes

This module contains routes and functions for handling discussions.
"""

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from deepsearcher import configuration
from deepsearcher.api.models import (
    RelatedType,
    DiscussCreateRequest,
    DiscussCreateResponse,
    DiscussPostRequest,
    DiscussPostResponse,
    DiscussAIReplyRequest,
    DiscussListRequest,
    DiscussListResponse,
    DiscussListEntity,
    ExceptionResponse,
    SortType,
    SummaryRequest,
    DepressCache,
)
from deepsearcher.api.rbase_util import (
    get_discuss_thread_by_request_hash,
    get_discuss_thread_by_uuid,
    save_discuss_thread_to_db,
    is_thread_has_summary,
    get_discuss_by_uuid,
    get_discuss_by_thread_uuid,
    save_discuss_to_db,
    update_discuss_thread_depth,
    get_discuss_history,
    get_mysql_pool,
    get_response_by_request_hash,
)
from deepsearcher.agent.discuss_agent import DiscussAgent
from deepsearcher.rbase.ai_models import (
    Discuss,
    DiscussThread,
    DiscussRole,
    AIResponseStatus,
    initialize_discuss_thread,
    initialize_ai_request_by_summary,
)
from .metadata import build_metadata

router = APIRouter()

@router.post(
    "/discuss_create",
    summary="Discussion Topic Creation API",
    description="""
    Create a new discussion topic or return existing topic UUID.
    
    - Supports author, topic, and paper related types
    - Uses request_hash and user_hash to check for existing topics
    - Returns topic UUID
    """,
)
async def api_create_discuss_thread(request: DiscussCreateRequest):
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
            has_summary = await is_thread_has_summary(result.id)
            return DiscussCreateResponse(
                code=0,
                message="success",
                thread_uuid=result.uuid,
                depth=result.depth,
                has_summary=has_summary
            )
        else:
            # If not exists, create new thread
            await save_discuss_thread_to_db(discuss_thread)
            return DiscussCreateResponse(
                code=0,
                message="success",
                thread_uuid=discuss_thread.uuid,
                depth=discuss_thread.depth,
                has_summary=False
            )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        )

@router.get(
    "/list_discuss",
    summary="List Discussion Topics API",
    description="""
    List discussion topics.
    
    - Supports author, topic, and paper related types
    - Optional cache usage
    - Supports streaming response
    """,
)
async def api_list_discuss(request: DiscussListRequest):
    """
    List discussion topics.

    Args:
        request (DiscussCreateRequest): Request parameters for listing discussion topics

    Returns:
        DiscussListResponse: Response containing discussion topics
    """
    try:
        # 1. 验证讨论话题是否存在
        thread = await get_discuss_thread_by_uuid(request.thread_uuid, user_hash=request.user_hash)
        if not thread:
            return JSONResponse(
                status_code=400,
                content=ExceptionResponse(
                    code=400, 
                    message=f"讨论话题不存在: {request.thread_uuid}"
                ).model_dump()
            )
        
        # 2. 从数据库获取讨论列表
        discuss_entities = []
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # 构建SQL查询
                sql = """
                SELECT * FROM discuss 
                WHERE thread_uuid = %s AND is_hidden = 0 AND status = %s
                """
                params = [request.thread_uuid, AIResponseStatus.FINISHED.value]
                
                # 添加深度和排序条件
                if request.sort == SortType.ASC:
                    sql += " AND depth >= %s ORDER BY depth ASC, created ASC LIMIT %s"
                else:
                    sql += " AND depth <= %s ORDER BY depth DESC, created DESC LIMIT %s"

                params.extend([request.from_depth, request.limit])
                
                # 执行查询
                await cursor.execute(sql, params)
                results = await cursor.fetchall()
                
                # 3. 构建响应数据
                for result in results:
                    # 创建实体对象
                    entity = DiscussListEntity(
                        uuid=result["uuid"],
                        depth=result["depth"],
                        content=result["content"],
                        created=int(result["created"].timestamp()),  # 将datetime转换为整数时间戳
                        role=result["role"],
                        is_summary=result["is_summary"],
                        user_hash=request.user_hash,
                        user_id=result["user_id"] if result["user_id"] else 0,
                        user_name=result["user_name"] if "user_name" in result else "",
                        user_avatar=result["user_avatar"] if "user_avatar" in result else ""
                    )
                    discuss_entities.append(entity)
                
                # 4. 返回响应
                return DiscussListResponse(
                    code=0,
                    message="success",
                    count=len(discuss_entities),
                    discuss_list=discuss_entities
                )
                
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        )

@router.post(
    "/discuss_post",
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
            await update_discuss_thread_depth(thread.uuid, discuss.depth, discuss.uuid)
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
    "/ai_reply",
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
            related_type=thread.related_type,
            thread_id=thread.id,
            thread_uuid=thread.uuid,
            reply_id=reply_discuss.id if reply_discuss else None,
            reply_uuid=reply_discuss.uuid if reply_discuss else None,
            depth=reply_discuss.depth + 1 if reply_discuss else 0,
            content="",  # Content will be updated during streaming generation
            role=DiscussRole.ASSISTANT,
            tokens={},
            usage={},
            status=AIResponseStatus.GENERATING,
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

def initialize_discuss_by_post_request(request: DiscussPostRequest, thread: DiscussThread, replyDiscuss: Discuss) -> Discuss:
    """
    Initialize a discuss object based on the post request.
    """
    discuss = Discuss(
        uuid="",
        related_type=thread.related_type,
        thread_id=thread.id,
        thread_uuid=thread.uuid,
        reply_id=replyDiscuss.id if replyDiscuss else None,
        reply_uuid=replyDiscuss.uuid if replyDiscuss else None,
        depth=replyDiscuss.depth + 1 if replyDiscuss else thread.depth + 1,
        content=request.content,
        role=DiscussRole.USER,
        tokens={},
        usage={},
        is_summary=0,
        status=AIResponseStatus.FINISHED,
        user_id=request.user_id if request.user_id else None,
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
            verbose=configuration.config.rbase_settings.get("verbose", False)
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
        ai_discuss.tokens["generating"] = []
        ai_discuss.status = AIResponseStatus.FINISHED
        if hasattr(discuss_agent, "usage"):
            ai_discuss.usage = discuss_agent.usage
        await save_discuss_to_db(ai_discuss)
        await update_discuss_thread_depth(thread.uuid, ai_discuss.depth, ai_discuss.uuid)
        
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
        has_summary = await is_thread_has_summary(thread.id)
        if has_summary:
            background_discuss = await get_discuss_by_thread_uuid(thread.uuid, is_summary=1)
            if background_discuss:
                return background_discuss.content

        if thread.related_type == RelatedType.CHANNEL or thread.related_type == RelatedType.COLUMN:
            channel_id = thread.params.get("channel_id", 0)
            if channel_id:
                # Try to get channel information from cached AI summary
                summary_request = SummaryRequest(
                    related_type=thread.related_type,
                    related_id=channel_id,
                    term_tree_node_ids=thread.params.get("term_tree_node_ids", []),
                    ver=thread.params.get("ver", 0),
                    depress_cache=DepressCache.DISABLE,
                    stream=False
                )
                metadata = await build_metadata(summary_request.related_type, 
                                                summary_request.related_id, 
                                                summary_request.term_tree_node_ids)
                ai_request = initialize_ai_request_by_summary(summary_request, metadata)
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