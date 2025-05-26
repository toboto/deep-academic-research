"""
Discussion Database Operations

This module contains database operations for discussions.
"""

import json
from datetime import datetime
from deepsearcher import configuration
from deepsearcher.db.async_mysql_connection import get_mysql_pool
from deepsearcher.tools import log
from deepsearcher.rbase.ai_models import (
    DiscussThread,
    Discuss,
    DiscussRole,
    RelatedType,
    AIResponseStatus,
    AIContentResponse,
)

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
                    depth=reply["depth"] + 1 if reply else thread["depth"] + 1,
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
                await save_discuss(discuss)
                await update_discuss_thread_depth(thread_uuid, discuss.depth, discuss.uuid)
    except Exception as e:
        raise Exception(f"Failed to update ai content to discuss: {e}")

async def update_discuss_thread_depth(thread_uuid: str, depth: int, discuss_uuid: str = None):
    """
    Update discuss thread depth
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await conn.begin()
                try:
                    # Update discuss thread depth
                    sql = "UPDATE discuss_thread SET depth = %s WHERE uuid = %s"
                    await cursor.execute(sql, (depth, thread_uuid))
                    
                    # If discuss_uuid is provided, update other discuss content status to deprecated
                    if discuss_uuid:
                        sql = "UPDATE discuss SET status = %s WHERE uuid <> %s AND thread_uuid = %s AND depth = %s"
                        await cursor.execute(sql, (AIResponseStatus.DEPRECATED.value, discuss_uuid, thread_uuid, depth))
                    
                    await conn.commit()
                except Exception as e:
                    # Rollback transaction on error
                    await conn.rollback()
                    raise e
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
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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

async def save_discuss_thread(discuss_thread: DiscussThread) -> int:
    """
    Save discussion thread to database

    Args:
        discuss_thread: DiscussThread object

    Returns:
        int: Inserted record ID
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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
    if not uuid:
        return None
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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

async def save_discuss(discuss: Discuss) -> int:
    """
    Save discussion content to database
    
    Args:
        discuss_content: Discussion content object
        
    Returns:
        int: Inserted record ID
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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
                        status = %s
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

async def get_discuss_in_thread(thread_uuid: str, discuss_uuid: str = None, **kwargs) -> Discuss:
    """
    Get discussion record by discuss UUID within the thread of thread_uuid.
    If is_summary is True, get the summary record. 
    If discuss_uuid is not None, get the specific record.
    
    Args:
        thread_uuid: Topic UUID
        discuss_uuid: Discussion UUID
        is_summary: Whether to get the summary record
    Returns:
        Discuss: Discussion content object, returns None if not found
    """
    if not kwargs and not discuss_uuid:
        return None
    
    is_summary = kwargs.get("is_summary", 0)
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = "SELECT * FROM discuss WHERE thread_uuid = %s AND is_hidden = 0 AND status = %s "
                params = [thread_uuid, AIResponseStatus.FINISHED.value]
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

async def get_discuss_thread_history(thread_id: int, reply_id: int, limit: int = 10, **kwargs) -> list:
    """
    Get discussion history records
    
    Args:
        thread_id: Discussion topic ID
        reply_id: Current reply ID
        limit: Limit on number of history records to retrieve
        
    Returns:
        list: History records list, sorted by time in ascending order, format as [{"role": "user|assistant", "content": "content"}]
    """
    role = kwargs.get("role", None)
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get history records before current discussion node
                params = [thread_id, AIResponseStatus.FINISHED.value]
                sql = """
                SELECT id, role, content FROM discuss 
                WHERE thread_id = %s AND is_hidden = 0 AND status = %s
                """
                if reply_id > 0:
                    sql += " AND id <= %s"
                    params.append(reply_id)
                if role:
                    sql += " AND role = %s"
                    params.append(role.value)

                sql += "\nORDER BY id DESC LIMIT %s"
                params.append(limit)
                await cursor.execute(sql, params)
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

async def list_discuss_in_thread(thread_uuid: str, from_depth: int, limit: int, sort_asc: bool = True) -> list[Discuss]:
    """
    获取指定讨论主题中的讨论内容列表
    
    Args:
        thread_uuid: 讨论主题UUID
        from_depth: 起始深度
        limit: 获取条数
        sort_asc: 是否按深度升序排序
        
    Returns:
        list[Discuss]: 讨论内容列表
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # 构建SQL查询
                sql = """
                SELECT * FROM discuss 
                WHERE thread_uuid = %s AND is_hidden = 0 AND status = %s
                """
                params = [thread_uuid, AIResponseStatus.FINISHED.value]
                
                # 添加深度和排序条件
                if sort_asc:
                    sql += " AND depth >= %s ORDER BY depth ASC, created ASC LIMIT %s"
                else:
                    sql += " AND depth <= %s ORDER BY depth DESC, created DESC LIMIT %s"

                params.extend([from_depth, limit])
                
                # 执行查询
                await cursor.execute(sql, params)
                results = await cursor.fetchall()
                
                # 转换结果
                discuss_list = []
                for result in results:
                    result["tokens"] = json.loads(result["tokens"]) if result["tokens"] else {}
                    result["usage"] = json.loads(result["usage"]) if result["usage"] else {}
                    result["role"] = DiscussRole(result["role"])
                    result["status"] = AIResponseStatus(result["status"])
                    discuss_list.append(Discuss(**result))
                
                return discuss_list
    except Exception as e:
        raise Exception(f"Failed to list discuss in thread: {e}") 