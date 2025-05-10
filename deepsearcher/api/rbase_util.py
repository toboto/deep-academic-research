import json
import hashlib
from datetime import datetime
from pydantic import BaseModel
from deepsearcher.rbase.ai_models import AIContentResponse, AIContentRequest, AIRequestStatus, AIResponseStatus
from deepsearcher import configuration
from deepsearcher.db.async_mysql_connection import get_mysql_pool, close_mysql_pool
from deepsearcher.rbase.ai_models import DiscussThread, RelatedType, Discuss


async def get_response_by_request_hash(request_hash: str) -> AIContentResponse:
    """
    根据请求hash获取响应内容

    Args:
        request_hash: 请求hash值

    Returns:
        AIContentResponse: 响应内容对象，如果未找到则返回None
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # 查询已完成的请求
                request_sql = """
                SELECT id FROM ai_content_request 
                WHERE request_hash = %s AND status = %s
                ORDER BY modified DESC LIMIT 1
                """
                await cursor.execute(request_sql, (request_hash, AIRequestStatus.FINISHED.value))
                request_result = await cursor.fetchone()
                
                if not request_result:
                    return None
                    
                request_id = request_result["id"]
                
                # 查询对应的响应
                response_sql = """
                SELECT * FROM ai_content_response 
                WHERE ai_request_id = %s
                ORDER BY modified DESC LIMIT 1
                """
                await cursor.execute(response_sql, (request_id,))
                response_result = await cursor.fetchone()
                
                if not response_result:
                    return None
                    
                # 处理双重编码的JSON字符串
                tokens_str = response_result["tokens"]
                usage_str = response_result["usage"]
                
                # 第一次解析：将字符串转换为JSON字符串
                tokens_json = json.loads(tokens_str) if tokens_str else "{}"
                usage_json = json.loads(usage_str) if usage_str else "{}"
                
                # 第二次解析：将JSON字符串转换为字典
                tokens_dict = json.loads(tokens_json) if isinstance(tokens_json, str) else tokens_json
                usage_dict = json.loads(usage_json) if isinstance(usage_json, str) else usage_json
                    
                # 构造响应对象
                return AIContentResponse(
                    id=response_result["id"],
                    ai_request_id=request_id,
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
    保存请求到数据库

    Args:
        request: AIContentRequest对象

    Returns:
        int: 插入记录的ID
    """
    if modified:
        request.modified = modified
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if request.id == 0:
                    # 插入请求记录
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
                    # 更新请求记录
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
    保存响应到数据库

    Args:
        response: AIContentResponse对象
    """
    if modified:
        response.modified = modified
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if response.id == 0:
                    # 插入响应记录
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
                    # 更新响应记录
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


async def get_discuss_thread_by_request_hash(request_hash: str, user_hash: str) -> DiscussThread:
    """
    根据请求hash和用户hash获取讨论线程

    Args:
        request_hash: 请求hash值
        user_hash: 用户hash值

    Returns:
        DiscussThread: 讨论线程对象，如果未找到则返回None
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

async def get_discuss_thread_by_id(thread_id: int) -> DiscussThread:
    """
    根据ID获取讨论线程

    Args:
        thread_id: 讨论线程ID

    Returns:
        DiscussThread: 讨论线程对象，如果未找到则返回None
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
    保存讨论线程到数据库

    Args:
        discuss_thread: DiscussThread对象

    Returns:
        int: 插入记录的ID
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
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
    except Exception as e:
        raise Exception(f"Failed to save discuss thread to db: {e}")

def get_request_hash(request: BaseModel) -> str:
    """
    计算请求的hash值
    
    Args:
        request: 请求对象
        
    Returns:
        str: 请求的hash值
    """
    # 将请求对象转换为字典
    request_dict = request.model_dump()
    
    # 移除不需要参与hash计算的字段
    if 'user_hash' in request_dict:
        del request_dict['user_hash']
    if 'user_id' in request_dict:
        del request_dict['user_id']
        
    # 将字典转换为JSON字符串
    request_json = json.dumps(request_dict, sort_keys=True)
    
    # 计算hash值
    return hashlib.md5(request_json.encode()).hexdigest()

async def get_discuss_thread_by_uuid(thread_uuid: str) -> DiscussThread:
    """
    根据话题UUID获取讨论线程
    
    Args:
        thread_uuid: 话题UUID
        
    Returns:
        DiscussThread: 讨论线程对象，如果未找到则返回None
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                SELECT * FROM discuss_thread WHERE uuid = %s AND is_hidden = 0
                """
                await cursor.execute(sql, (thread_uuid,))
                result = await cursor.fetchone()
                if not result:
                    return None
                result["params"] = json.loads(result["params"]) if result["params"] else {}
                return DiscussThread(**result)
    except Exception as e:
        raise Exception(f"获取讨论线程失败: {e}")

async def get_discuss_by_uuid(uuid: str) -> Discuss:
    """
    根据内容UUID获取讨论内容
    
    Args:
        content_uuid: 内容UUID
        
    Returns:
        DiscussContent: 讨论内容对象，如果未找到则返回None
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
                result["tokens"] = json.loads(result["tokens"]) if result["tokens"] else {}
                result["usage"] = json.loads(result["usage"]) if result["usage"] else {}
                result["status"] = AIResponseStatus(result["status"])
                return Discuss(**result)
    except Exception as e:
        raise Exception(f"获取讨论内容失败: {e}")

async def save_discuss_to_db(discuss: Discuss) -> int:
    """
    保存讨论内容到数据库
    
    Args:
        discuss_content: 讨论内容对象
        
    Returns:
        int: 插入记录的ID
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                INSERT INTO discuss (
                    uuid, thread_id, thread_uuid, reply_id, reply_uuid, depth, content, 
                    role, tokens, `usage`, user_id, is_hidden, `like`, trample, status, 
                    created, modified
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                await cursor.execute(sql, (
                    discuss.uuid,
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
                    discuss.status.value,
                    discuss.created,
                    discuss.modified
                ))
                return cursor.lastrowid
    except Exception as e:
        raise Exception(f"保存讨论内容失败: {e}")

async def get_discuss_by_thread_uuid(thread_uuid: str) -> list:
    """
    根据话题UUID获取所有讨论内容
    
    Args:
        thread_uuid: 话题UUID
        
    Returns:
        list: 讨论内容对象列表
    """
    pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = """
                SELECT * FROM discuss 
                WHERE thread_uuid = %s AND is_hidden = 0 AND status = %s
                ORDER BY depth ASC
                """
                await cursor.execute(sql, (thread_uuid, AIResponseStatus.FINISHED.value))
                results = await cursor.fetchall()
                rts = []
                for result in results:
                    result["tokens"] = json.loads(result["tokens"]) if result["tokens"] else {}
                    result["usage"] = json.loads(result["usage"]) if result["usage"] else {}
                    result["status"] = AIResponseStatus(result["status"])
                    rts.append(Discuss(**result))
                return rts
    except Exception as e:
        raise Exception(f"获取讨论内容列表失败: {e}")
