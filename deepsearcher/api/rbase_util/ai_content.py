"""
AI Content Database Operations

This module contains database operations for AI content requests and responses.
"""

import json
from datetime import datetime
from deepsearcher import configuration
from deepsearcher.db.async_mysql_connection import get_mysql_pool
from deepsearcher.rbase.ai_models import (
    AIContentRequest,
    AIContentResponse,
    AIRequestStatus,
    AIResponseStatus,
)

async def get_response_by_request_hash(request_hash: str) -> AIContentResponse:
    """
    Get response content by request hash

    Args:
        request_hash: Request hash value

    Returns:
        AIContentResponse: Response content object, returns None if not found
    """
    cache_days = configuration.config.rbase_settings.get("api", {}).get("summary_cache_days", 30)
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Query corresponding response
                response_sql = """
                SELECT resp.* FROM ai_content_response resp 
                    LEFT JOIN ai_content_request req ON resp.ai_request_id = req.id
                    WHERE req.request_hash = %s AND req.`status` = %s AND resp.`status` = %s
                          AND resp.created > DATE_SUB(CURDATE(), INTERVAL %s DAY)
                    ORDER BY resp.modified DESC LIMIT 1
                """
                await cursor.execute(response_sql, (
                    request_hash, 
                    AIRequestStatus.FINISHED.value, 
                    AIResponseStatus.FINISHED.value, 
                    cache_days
                    ))
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

    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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

    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
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