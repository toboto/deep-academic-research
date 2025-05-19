"""
Database operation utility module
"""

import json
import hashlib
from datetime import datetime
from pydantic import BaseModel
from deepsearcher import configuration
from deepsearcher.db.async_mysql_connection import get_mysql_pool, close_mysql_pool
from deepsearcher.tools import log
from deepsearcher.rbase.ai_models import (
    DiscussThread, 
    RelatedType, 
    Discuss, 
    DiscussRole,
    AIContentRequest, 
    AIContentResponse, 
    AIRequestStatus, 
    AIResponseStatus,
    TermTreeNode
)


async def get_response_by_request_hash(request_hash: str) -> AIContentResponse:
    """
    Get response content by request hash

    Args:
        request_hash: Request hash value

    Returns:
        AIContentResponse: Response content object, returns None if not found
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Query corresponding response
                response_sql = """
                SELECT resp.* FROM ai_content_response resp 
                    LEFT JOIN ai_content_request req ON resp.ai_request_id = req.id
                    WHERE req.request_hash = %s and req.`status` = %s
                    ORDER BY resp.modified DESC LIMIT 1
                """
                await cursor.execute(response_sql, (request_hash, AIRequestStatus.FINISHED.value))
                response_result = await cursor.fetchone()
                
                if not response_result:
                    return None
                
                update_hit_cnt_sql = """
                UPDATE ai_content_response SET cache_hit_cnt = cache_hit_cnt + 1
                WHERE id = %s
                """
                await cursor.execute(update_hit_cnt_sql, (response_result["id"],))
                    
                # Handle double-encoded JSON strings
                tokens_str = response_result["tokens"]
                usage_str = response_result["usage"]
                
                # First parse: Convert string to JSON string
                tokens_json = json.loads(tokens_str) if tokens_str else "{}"
                usage_json = json.loads(usage_str) if usage_str else "{}"
                
                # Second parse: Convert JSON string to dictionary
                tokens_dict = json.loads(tokens_json) if isinstance(tokens_json, str) else tokens_json
                usage_dict = json.loads(usage_json) if isinstance(usage_json, str) else usage_json
                    
                # Construct response object
                return AIContentResponse(
                    id=response_result["id"],
                    ai_request_id=response_result["ai_request_id"],
                    is_generating=response_result["is_generating"],
                    content=response_result["content"],
                    tokens=tokens_dict,
                    usage=usage_dict,
                    cache_hit_cnt=response_result["cache_hit_cnt"],
                    status=AIResponseStatus(response_result["status"]),
                    created=response_result["created"],
                    modified=response_result["modified"]
                )
    except Exception as e:
        raise Exception(f"Failed to get response by request hash: {e}")


async def save_request_to_db(request: AIContentRequest, modified: datetime = datetime.now()) -> int:
    """
    Save request to database

    Args:
        request: AIContentRequest object

    Returns:
        int: Inserted record ID
    """
    if modified:
        request.modified = modified
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if request.id == 0:
                    # Insert request record
                    sql = """
                    INSERT INTO ai_content_request (
                        content_type, is_stream_response, query, params,
                        request_hash, status, created, modified
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    await cursor.execute(sql, (
                        request.content_type.value,
                        request.is_stream_response.value,
                        request.query,
                        json.dumps(request.params),
                        request.request_hash,
                        request.status.value,
                        request.created,
                        request.modified
                    ))
                    return cursor.lastrowid
                else:
                    # Update request record
                    sql = """
                    UPDATE ai_content_request SET
                        status = %s,
                        modified = %s
                    WHERE id = %s
                    """
                    await cursor.execute(sql, (
                        request.status.value,
                        request.modified,
                        request.id
                    ))
                    return request.id
    except Exception as e:
        raise Exception(f"Failed to save request to db: {e}")


async def save_response_to_db(response: AIContentResponse, modified: datetime = datetime.now()) -> int:
    """
    Save response to database

    Args:
        response: AIContentResponse object

    Returns:
        int: Inserted record ID
    """
    if modified:
        response.modified = modified
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if response.id == 0:
                    # Insert response record
                    sql = """
                    INSERT INTO ai_content_response (
                        ai_request_id, is_generating, content, tokens, 
                        `usage`, cache_hit_cnt, status, created, modified
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """
                    await cursor.execute(sql, (
                        response.ai_request_id,
                        response.is_generating,
                        response.content,
                        json.dumps(response.tokens),
                        json.dumps(response.usage),
                        response.cache_hit_cnt,
                        response.status.value,
                        response.created,
                        response.modified
                    ))
                    return cursor.lastrowid
                else:
                    # Update response record
                    sql = """
                    UPDATE ai_content_response SET
                        is_generating = %s,
                        content = %s,
                        tokens = %s,
                        `usage` = %s,
                        cache_hit_cnt = %s,
                        status = %s,
                        modified = %s
                    WHERE id = %s
                    """
                    await cursor.execute(sql, (
                        response.is_generating,
                        response.content,
                        json.dumps(response.tokens),
                        json.dumps(response.usage),
                        response.cache_hit_cnt,
                        response.status.value,
                        response.modified,
                        response.id
                    ))
                    return response.id
    except Exception as e:
        raise Exception(f"Failed to save response to db: {e}")

async def update_ai_content_to_discuss(response: AIContentResponse, thread_uuid: str, reply_uuid: str):
    """
    Update AI content to discuss

    Args:
        response: AIContentResponse object
        thread_uuid: Thread UUID
        reply_uuid: Reply UUID
    """
    if not thread_uuid:
        return

    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = "SELECT * FROM discuss_thread WHERE uuid = %s"
                await cursor.execute(sql, (thread_uuid,))
                thread = await cursor.fetchone()
                if not thread:
                    raise Exception(f"Failed to get discuss thread by uuid: {thread_uuid}")
                if thread["depth"] > 0 and reply_uuid:
                    sql = "SELECT * FROM discuss WHERE uuid = %s"
                    await cursor.execute(sql, (reply_uuid,))
                    reply = await cursor.fetchone()
                    if not reply:
                        raise Exception(f"Failed to get discuss by uuid: {reply_uuid}")
                else:
                    reply = None

                discuss = Discuss(
                    related_type=RelatedType(thread["related_type"]),
                    thread_id=thread["id"],
                    thread_uuid=thread["uuid"],
                    reply_id=reply["id"] if reply else None,
                    reply_uuid=reply["uuid"] if reply else None,
                    depth=reply["depth"] if reply else 1,
                    content=response.content,
                    tokens=response.tokens,
                    usage=response.usage,
                    is_summary=1,
                    role=DiscussRole.ASSISTANT,
                    status=AIResponseStatus.FINISHED,
                    created=datetime.now(),
                    modified=datetime.now()
                )
                discuss.create_uuid()
                await save_discuss_to_db(discuss)
    except Exception as e:
        raise Exception(f"Failed to update ai content to discuss: {e}")

async def update_discuss_thread_depth(thread_uuid: str, depth: int):
    """
    Update discuss thread depth
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = "UPDATE discuss_thread SET depth = %s WHERE uuid = %s"
                await cursor.execute(sql, (depth, thread_uuid))
    except Exception as e:
        raise Exception(f"Failed to update discuss thread depth: {e}")

async def get_discuss_thread_by_request_hash(request_hash: str, user_hash: str) -> DiscussThread:
    """
    Get discussion thread by request hash and user hash

    Args:
        request_hash: Request hash value
        user_hash: User hash value

    Returns:
        DiscussThread: Discussion thread object, returns None if not found
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                SELECT * FROM discuss_thread WHERE request_hash = %s AND user_hash = %s
                """
                await cursor.execute(sql, (request_hash, user_hash))
                result = await cursor.fetchone()
                if not result:
                    return None
                result["params"] = json.loads(result["params"]) if result["params"] else {}
                result["related_type"] = RelatedType(result["related_type"])
                return DiscussThread(**result)
    except Exception as e:
        raise Exception(f"Failed to get discuss thread by request hash: {e}")
    
async def is_thread_has_summary(thread_id: int) -> bool:
    """
    Check if the discussion thread has a summary
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                SELECT COUNT(*) as cnt FROM discuss WHERE thread_id = %s AND is_summary = 1
                """
                await cursor.execute(sql, (thread_id,))
                result = await cursor.fetchone()
                return result["cnt"] > 0
    except Exception as e:
        raise Exception(f"Failed to check if thread has summary: {e}")

async def get_discuss_thread_by_id(thread_id: int) -> DiscussThread:
    """
    Get discussion thread by ID

    Args:
        thread_id: Discussion thread ID

    Returns:
        DiscussThread: Discussion thread object, returns None if not found
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                SELECT * FROM discuss_thread WHERE id = %s AND is_hidden = 0
                """
                await cursor.execute(sql, (thread_id,))
                result = await cursor.fetchone()
                if not result:
                    return None
                result["params"] = json.loads(result["params"]) if result["params"] else {}
                result["related_type"] = RelatedType(result["related_type"])
                return DiscussThread(**result)
    except Exception as e:
        raise Exception(f"Failed to get discuss thread by id: {e}")

async def save_discuss_thread_to_db(discuss_thread: DiscussThread) -> int:
    """
    Save discussion thread to database

    Args:
        discuss_thread: DiscussThread object

    Returns:
        int: Inserted record ID
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if discuss_thread.id == 0:
                    sql = """
                    INSERT INTO discuss_thread (uuid, related_type, params, request_hash, user_hash, user_id, depth, background, is_hidden, created, modified)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    await cursor.execute(sql, (
                        discuss_thread.uuid,
                        discuss_thread.related_type.value,
                        json.dumps(discuss_thread.params),
                        discuss_thread.request_hash,
                        discuss_thread.user_hash,
                        discuss_thread.user_id,
                        discuss_thread.depth,
                        discuss_thread.background,
                        discuss_thread.is_hidden,
                        discuss_thread.created,
                        discuss_thread.modified
                    ))
                    return cursor.lastrowid
                else:
                    sql = """
                    UPDATE discuss_thread SET
                        relate_type = %s,
                        params = %s,
                        request_hash = %s,
                        user_hash = %s,
                        user_id = %s,
                        depth = %s,
                        background = %s,
                        is_hidden = %s,
                    WHERE id = %s
                    """
                    await cursor.execute(sql, (
                        discuss_thread.relate_type.value,
                        json.dumps(discuss_thread.params),
                        discuss_thread.request_hash,
                        discuss_thread.user_hash,
                        discuss_thread.user_id,
                        discuss_thread.depth,
                        discuss_thread.background,
                        discuss_thread.is_hidden,
                        discuss_thread.id
                    ))
                    return discuss_thread.id
    except Exception as e:
        raise Exception(f"Failed to save discuss thread to db: {e}")

def get_request_hash(request: BaseModel) -> str:
    """
    Calculate request hash value
    
    Args:
        request: Request object
        
    Returns:
        str: Request hash value
    """
    # Convert request object to dictionary
    request_dict = request.model_dump()
    
    # Remove fields that should not participate in hash calculation
    if 'user_hash' in request_dict:
        del request_dict['user_hash']
    if 'user_id' in request_dict:
        del request_dict['user_id']
        
    # Convert dictionary to JSON string
    request_json = json.dumps(request_dict, sort_keys=True)
    
    # Calculate hash value
    return hashlib.md5(request_json.encode()).hexdigest()

async def get_discuss_thread_by_uuid(thread_uuid: str, **kwargs) -> DiscussThread:
    """
    Get discussion thread by topic UUID
    
    Args:
        thread_uuid: Topic UUID
        
    Returns:
        DiscussThread: Discussion thread object, returns None if not found
    """
    user_hash = kwargs.get("user_hash", None)
    user_id = kwargs.get("user_id", None)
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                params = [thread_uuid]
                sql = """
                SELECT * FROM discuss_thread WHERE uuid = %s AND is_hidden = 0
                """
                if user_hash:
                    sql += " AND user_hash = %s"
                    params.append(user_hash)
                if user_id:
                    sql += " AND user_id = %s"
                    params.append(user_id)

                await cursor.execute(sql, params)
                result = await cursor.fetchone()
                if not result:
                    return None
                result["related_type"] = RelatedType(result["related_type"])
                result["params"] = json.loads(result["params"]) if result["params"] else {}
                return DiscussThread(**result)
    except Exception as e:
        raise Exception(f"Failed to get discuss thread: {e}")

async def get_discuss_by_uuid(uuid: str) -> Discuss:
    """
    Get discussion content by content UUID
    
    Args:
        content_uuid: Content UUID
        
    Returns:
        DiscussContent: Discussion content object, returns None if not found
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                SELECT * FROM discuss WHERE uuid = %s AND is_hidden = 0 AND status = %s
                """
                await cursor.execute(sql, (uuid, AIResponseStatus.FINISHED.value))
                result = await cursor.fetchone()
                if not result:
                    return None
                result["related_type"] = RelatedType(result["related_type"])
                result["tokens"] = json.loads(result["tokens"]) if result["tokens"] else {}
                result["usage"] = json.loads(result["usage"]) if result["usage"] else {}
                result["role"] = DiscussRole(result["role"])
                result["status"] = AIResponseStatus(result["status"])
                return Discuss(**result)
    except Exception as e:
        raise Exception(f"Failed to get discuss content: {e}")

async def save_discuss_to_db(discuss: Discuss) -> int:
    """
    Save discussion content to database
    
    Args:
        discuss_content: Discussion content object
        
    Returns:
        int: Inserted record ID
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if discuss.id == 0:
                    sql = """
                    INSERT INTO discuss (
                        uuid, related_type, thread_id, thread_uuid, reply_id, reply_uuid, depth, 
                        content, role, tokens, `usage`, user_id, is_hidden, `like`, trample, 
                        is_summary, status, created, modified
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, 
                        %s, %s, %s, %s, %s, %s, %s, %s, 
                        %s, %s, %s, %s
                    )
                    """
                    await cursor.execute(sql, (
                        discuss.uuid,
                        discuss.related_type.value,
                        discuss.thread_id,
                        discuss.thread_uuid,
                        discuss.reply_id,
                        discuss.reply_uuid,
                        discuss.depth,
                        discuss.content,
                        discuss.role.value,
                        json.dumps(discuss.tokens),
                        json.dumps(discuss.usage),
                        discuss.user_id,
                        discuss.is_hidden,
                        discuss.like,
                        discuss.trample,
                        discuss.is_summary,
                        discuss.status.value,
                        discuss.created,
                        discuss.modified
                    ))
                    return cursor.lastrowid
                else:
                    sql = """
                    UPDATE discuss SET
                        content = %s,
                        role = %s,
                        tokens = %s,
                        `usage` = %s,
                        is_hidden = %s,
                        `like` = %s,
                        trample = %s,
                        is_summary = %s,
                        status = %s,
                    WHERE id = %s
                    """
                    await cursor.execute(sql, (
                        discuss.content,
                        discuss.role.value,
                        json.dumps(discuss.tokens),
                        json.dumps(discuss.usage),
                        discuss.is_hidden,
                        discuss.like,
                        discuss.trample,
                        discuss.is_summary,
                        discuss.status.value,
                        discuss.id
                    ))
                    return discuss.id
    except Exception as e:
        raise Exception(f"Failed to save discuss content: {e}")

async def get_discuss_by_thread_uuid(thread_uuid: str, discuss_uuid: str = None, **kwargs) -> Discuss:
    """
    Get discussion record by topic UUID, if is_summary is True, get the summary record, if discuss_uuid is not None, get the specific record
    
    Args:
        thread_uuid: Topic UUID
        
    Returns:
        list: List of discussion content objects
    """
    if not kwargs and not discuss_uuid:
        return None
    
    is_summary = kwargs.get("is_summary", 0)
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = "SELECT * FROM discuss WHERE thread_uuid = %s AND is_hidden = 0 AND status = %s "
                params = [thread_uuid, AIResponseStatus.FINISHED.value, discuss_uuid]
                if is_summary:
                    sql += "AND is_summary = 1"
                if discuss_uuid:
                    sql += "AND uuid = %s"
                    params.append(discuss_uuid)
                await cursor.execute(sql, params)
                result = await cursor.fetchone()
                if not result:
                    return None
                result["tokens"] = json.loads(result["tokens"]) if result["tokens"] else {}
                result["usage"] = json.loads(result["usage"]) if result["usage"] else {}
                result["status"] = AIResponseStatus(result["status"])
                return Discuss(**result)
    except Exception as e:
        raise Exception(f"Failed to get discuss content list: {e}")

async def get_discuss_history(thread_id: int, reply_id: int, limit: int = 10) -> list:
    """
    Get discussion history records
    
    Args:
        thread_id: Discussion topic ID
        reply_id: Current reply ID
        limit: Limit on number of history records to retrieve
        
    Returns:
        list: History records list, sorted by time in ascending order, format as [{"role": "user|assistant", "content": "content"}]
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get history records before current discussion node
                sql = """
                SELECT id, role, content FROM discuss 
                WHERE thread_id = %s AND id <= %s AND is_hidden = 0 AND status = %s
                ORDER BY id DESC LIMIT %s
                """
                await cursor.execute(sql, (thread_id, reply_id, AIResponseStatus.FINISHED.value, limit))
                results = await cursor.fetchall()
                
                # Convert format and sort by time
                history = []
                for result in sorted(results, key=lambda x: x["id"]):
                    history.append({
                        "role": result["role"],
                        "content": result["content"]
                    })
                
                return history
    except Exception as e:
        log.error(f"Failed to get discussion history records: {e}")
        return []


async def get_term_tree_nodes(term_tree_node_ids: list[int]) -> list[TermTreeNode]:
    """
    Get term tree nodes
    """
    if not term_tree_node_ids:
        return []

    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                placeholders = ', '.join(['%s'] * len(term_tree_node_ids))
                sql = """
                SELECT * FROM term_tree_node WHERE id IN ({placeholders})
                """
                sql = sql.format(placeholders=placeholders)
                await cursor.execute(sql, term_tree_node_ids)
                results = await cursor.fetchall()
                return [TermTreeNode(**result) for result in results]
    except Exception as e:
        raise Exception(f"Failed to get term tree nodes: {e}")
