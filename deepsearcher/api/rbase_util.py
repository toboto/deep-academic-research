import json
from datetime import datetime
from deepsearcher.api.models import AIContentResponse, AIContentRequest, AIRequestStatus, AIResponseStatus
from deepsearcher import configuration
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.db.mysql_connection import get_mysql_connection, close_mysql_connection


def get_response_by_request_hash(request_hash: str) -> AIContentResponse:
    """
    根据请求hash获取响应内容

    Args:
        request_hash: 请求hash值

    Returns:
        AIContentResponse: 响应内容对象，如果未找到则返回None
    """
    conn = get_mysql_connection(configuration.config.rbase_settings.get("database"))
    try:
        with conn.cursor() as cursor:
            # 查询已完成的请求
            request_sql = """
            SELECT id FROM ai_content_request 
            WHERE request_hash = %s AND status = %s
            ORDER BY modified DESC LIMIT 1
            """
            cursor.execute(request_sql, (request_hash, AIRequestStatus.FINISHED.value))
            request_result = cursor.fetchone()
            
            if not request_result:
                return None
                
            request_id = request_result["id"]
            
            # 查询对应的响应
            response_sql = """
            SELECT * FROM ai_content_response 
            WHERE ai_request_id = %s
            ORDER BY modified DESC LIMIT 1
            """
            cursor.execute(response_sql, (request_id,))
            response_result = cursor.fetchone()
            
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


def save_request_to_db(request: AIContentRequest, modified: datetime = datetime.now()) -> int:
    """
    保存请求到数据库

    Args:
        request: AIContentRequest对象

    Returns:
        int: 插入记录的ID
    """
    if modified:
        request.modified = modified
    conn = get_mysql_connection(configuration.config.rbase_settings.get("database"))
    try:
        with conn.cursor() as cursor:
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
                cursor.execute(sql, (
                    request.content_type.value,
                    request.is_stream_response.value,
                    request.query,
                    json.dumps(request.params),
                    request.request_hash,
                    request.status.value,
                    request.created,
                    request.modified
                ))
                conn.commit()
                return cursor.lastrowid
            else:
                # 更新请求记录
                sql = """
                UPDATE ai_content_request SET
                    status = %s,
                    modified = %s
                WHERE id = %s
                """
                cursor.execute(sql, (
                    request.status.value,
                    request.modified,
                    request.id
                ))
                conn.commit()
                return request.id
    except Exception as e:
        raise Exception(f"Failed to save request to db: {e}")


def save_response_to_db(response: AIContentResponse, modified: datetime = datetime.now()) -> int:
    """
    保存响应到数据库

    Args:
        response: AIContentResponse对象
    """
    if modified:
        response.modified = modified
    conn = get_mysql_connection(configuration.config.rbase_settings.get("database"))
    try:
        with conn.cursor() as cursor:
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
                cursor.execute(sql, (
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
                conn.commit()
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
                cursor.execute(sql, (
                    response.is_generating,
                    response.content,
                    json.dumps(response.tokens),
                    json.dumps(response.usage),
                    response.cache_hit_cnt,
                    response.status.value,
                    response.modified,
                    response.id
                ))
                conn.commit()
                return response.id
    except Exception as e:
        raise Exception(f"Failed to save response to db: {e}")
