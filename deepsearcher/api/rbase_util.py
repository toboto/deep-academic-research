import os
from deepsearcher.api.models import AIContentResponse, AIContentRequest, AIRequestStatus, AIResponseStatus
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.db.mysql_connection import get_mysql_connection, close_mysql_connection

current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file = os.path.join(current_dir, "..", "config.rbase.yaml")

# 从YAML文件加载配置
config = Configuration(yaml_file)

# 应用配置，使其在全局生效
init_config(config)


def get_response_by_request_hash(request_hash: str) -> AIContentResponse:
    """
    根据请求hash获取响应内容

    Args:
        request_hash: 请求hash值

    Returns:
        AIContentResponse: 响应内容对象，如果未找到则返回None
    """
    conn = get_mysql_connection(config.rbase_settings.get("database"))
    try:
        with conn.cursor() as cursor:
            # 查询已完成的请求
            request_sql = """
            SELECT id FROM ai_content_request 
            WHERE request_hash = %s AND status = %s
            ORDER BY modified DESC LIMIT 1
            """
            cursor.execute(request_sql, (request_hash, AIRequestStatus.FINISH_REQ))
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
                
            # 构造响应对象
            return AIContentResponse(
                id=response_result["id"],
                ai_request_id=request_id,
                is_generating=response_result["is_generating"],
                content=response_result["content"],
                tokens=response_result["tokens"],
                usage=response_result["usage"],
                cache_hit_cnt=response_result["cache_hit_cnt"],
                cache_miss_cnt=response_result["cache_miss_cnt"],
                status=response_result["status"],
                created=response_result["created"],
                modified=response_result["modified"]
            )
    finally:
        close_mysql_connection()


def save_request_to_db(request: AIContentRequest) -> int:
    """
    保存请求到数据库

    Args:
        request: AIContentRequest对象

    Returns:
        int: 插入记录的ID
    """
    conn = get_mysql_connection(config.rbase_settings.get("database"))
    try:
        with conn.cursor() as cursor:
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
                request.content_type,
                request.is_stream_response,
                request.query,
                request.params,
                request.request_hash,
                request.status,
                request.created,
                request.modified
            ))
            conn.commit()
            return cursor.lastrowid
    finally:
        close_mysql_connection()


