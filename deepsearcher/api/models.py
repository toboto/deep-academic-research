"""
API数据模型定义

本模块定义了FastAPI接口的请求和响应数据结构。
"""

import hashlib
import json
from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class RelatedType(int, Enum):
    """关联类型枚举"""
    CHANNEL = 1  # 频道
    COLUMN = 2   # 栏目
    ARTICLE = 3   # 文章


class DepressCache(int, Enum):
    """缓存抑制枚举"""
    ENABLE = 0  # 启用缓存
    DISABLE = 1 # 禁用缓存


class SummaryRequest(BaseModel):
    """AI概述接口请求模型"""
    related_type: RelatedType = Field(
        ...,
        description="关联类型：1-频道，2-栏目，3-文章"
    )
    related_id: Optional[int] = Field(
        None,
        description="关联ID，可选"
    )
    term_ids: Optional[List[int]] = Field(
        None,
        description="关键词ID列表，可选"
    )
    ver: int = Field(
        ...,
        description="版本号"
    )
    depress_cache: DepressCache = Field(
        ...,
        description="缓存抑制：0-启用缓存，1-禁用缓存"
    )
    stream: bool = Field(
        ...,
        description="是否使用流式响应"
    )


class SummaryResponse(BaseModel):
    """AI概述接口响应模型"""
    code: int = Field(
        ...,
        description="响应码：0-成功，非0-失败"
    )
    message: str = Field(
        ...,
        description="响应消息"
    )
    id: Optional[str] = Field(
        None,
        description="ID"
    )
    content: Optional[str] = Field(
        None,
        description="内容"
    )
    created: Optional[int] = Field(
        ...,
        description="创建时间"
    )
    model: Optional[str] = Field(
        None,
        description="模型"
    )
    object: Optional[str] = Field(
        None,
        description="对象"
    )
    choices: Optional[List[dict]] = Field(
        None,
        description="回答选项"
    )

    def setContent(self, content: str):
        self.content = content
        self.created = int(datetime.now().timestamp())
        self.id = f"chatcmpl-{self.created}"
        self.model = "rbase-rag"
        self.object = "chat.completion"
        self.choices = [
            {
                "index": 0,
                "message": {
                    "content": content,
                    "role": "assistant",
                },
                "finish_reason": "stop"
            }
        ]

class QuestionRequest(BaseModel):
    """AI推荐问题接口请求模型"""
    related_type: RelatedType = Field(
        ...,
        description="关联类型：1-作者，2-主题，3-论文"
    )
    related_id: Optional[int] = Field(
        None,
        description="关联ID，可选"
    )
    term_ids: Optional[List[int]] = Field(
        None,
        description="关键词ID列表，可选"
    )
    ver: int = Field(
        ...,
        description="版本号"
    )
    depress_cache: DepressCache = Field(
        ...,
        description="缓存抑制：0-启用缓存，1-禁用缓存"
    )
    count: int = Field(
        ...,
        description="问题数量"
    )


class QuestionResponse(BaseModel):
    """AI推荐问题接口响应模型"""
    code: int = Field(
        ...,
        description="响应码：0-成功，非0-失败"
    )
    message: str = Field(
        ...,
        description="响应消息"
    ) 
    questions: Optional[List[str]] = Field(
        None,
        description="问题列表"
    )

    def setQuestions(self, content: str):
        # 将content按换行符分割，得到问题列表
        questions = [q for q in content.strip().split('\n') if q]
        self.questions = questions

class AIContentType(int, Enum):
    """AI内容类型枚举"""
    LONG_SUMMARY = 1  # 长综述
    SHORT_SUMMARY = 2   # 短综述
    DISCUSSION = 10   # 讨论
    RECOMMEND_READ = 20   # 推荐阅读
    ASSOCIATED_QUESTION = 30   # 关联提问

class StreamResponse(int, Enum):
    """流式响应枚举"""
    DENY = 0  # 禁用流式响应
    ALLOW = 1 # 启用流式响应

class AIRequestStatus(int, Enum):
    """AI请求状态枚举"""
    START_REQ = 1 # 开始请求
    RECV_REQ = 2 # 收到请求
    HANDLING_REQ= 3 # 处理请求
    FINISHED = 10 # 已完成
    DEPRECATED = 100 # 已废弃

class AIContentRequest(BaseModel):
    """AI内容接口请求模型"""
    id: int = Field(
        ...,
        description="ID"
    )
    content_type: AIContentType = Field(
        ...,
        description="内容类型：1-长综述，2-短综述，10-讨论，20-推荐阅读，30-关联提问"
    )
    is_stream_response: StreamResponse = Field(
        ...,
        description="是否使用流式响应：0-禁用流式响应，1-启用流式响应"
    )
    query: str = Field(
        ...,
        description="查询内容"
    )
    params: dict = Field(
        ...,
        description="查询参数"
    )
    request_hash: str = Field(
        ...,
        description="请求hash"
    )
    status: AIRequestStatus = Field(
        ...,
        description="状态"
    )
    created: datetime = Field(
        ...,
        description="创建时间"
    )
    modified: datetime = Field(
        ...,
        description="更新时间"
    )

    def hash(self) -> str:
        # 综合query、params和content_type计算hash值
        hash_str = f"{self.content_type}_{self.query}_{json.dumps(self.params, sort_keys=True)}"
        self.request_hash = hashlib.md5(hash_str.encode()).hexdigest()

class AIResponseStatus(int, Enum):
    """AI响应状态枚举"""
    GENERATING = 1 # 生成中
    FINISHED = 10 # 已完成
    DEPRECATED = 100 # 已废弃

class AIContentResponse(BaseModel):
    """AI内容接口响应模型"""
    id: int = Field(
        ...,
        description="ID"
    )
    ai_request_id: int = Field(
        ...,
        description="AI请求ID"
    )
    is_generating: int = Field(
        ...,
        description="是否正在生成：0-不在生成，1-正在生成"
    )
    content: str = Field(
        ...,
        description="内容"
    )
    tokens: dict = Field(
        ...,
        description="令牌"
    )
    usage: dict = Field(
        ...,
        description="使用情况"
    )
    cache_hit_cnt: int = Field(
        ...,
        description="缓存命中次数"
    )
    status: AIResponseStatus = Field(
        ...,
        description="状态"
    )
    created: datetime = Field(
        ...,
        description="创建时间"
    )
    modified: datetime = Field(
        ...,
        description="更新时间"
    )


def initialize_ai_request_by_summary(request: SummaryRequest):
    """
    Initialize an AIContentRequest object from a SummaryRequest.
    
    Args:
        request (SummaryRequest): The source summary request
        
    Returns:
        AIContentRequest: A new AI content request object initialized with values from the summary request
    """
    ai_request = AIContentRequest(
        id=0,
        content_type=AIContentType.SHORT_SUMMARY,
        is_stream_response=StreamResponse.ALLOW if request.stream else StreamResponse.DENY,
        query=_create_query_by_summary_request(request, AIContentType.SHORT_SUMMARY),
        params=_create_params_by_summary_request(request),
        request_hash="",
        status=AIRequestStatus.START_REQ,
        created=datetime.now(),
        modified=datetime.now()
    )
    ai_request.hash()
    return ai_request

def initialize_ai_request_by_question(request: QuestionRequest):
    """
    Initialize an AIContentRequest object from a QuestionRequest.
    
    Args:
        request (QuestionRequest): The source question request
        
    Returns:
        AIContentRequest: A new AI content request object initialized with values from the question request
    """
    ai_request = AIContentRequest(
        id=0,
        content_type=AIContentType.ASSOCIATED_QUESTION,
        is_stream_response=StreamResponse.DENY,
        query=_create_query_by_question_request(request),
        params=_create_params_by_question_request(request),
        request_hash="",
        status=AIRequestStatus.START_REQ,
        created=datetime.now(),
        modified=datetime.now()
    )
    ai_request.hash()
    return ai_request

def initialize_ai_content_response(request: SummaryRequest, ai_content_request_id: int):
    """
    Initialize an AIContentResponse object from a SummaryRequest.
    
    Args:
        request (SummaryRequest): The source summary request
        
    Returns:
        AIContentResponse: A new AI content response object initialized with default values
    """
    ai_response = AIContentResponse(
        id=0,
        ai_request_id=ai_content_request_id,
        is_generating=0,
        content="",
        tokens={"generating": []},
        usage={},
        cache_hit_cnt=0,
        status=AIResponseStatus.GENERATING,
        created=datetime.now(),
        modified=datetime.now()
    )
    return ai_response

def _create_query_by_summary_request(request: SummaryRequest, content_type: AIContentType) -> str:
    """
    Create a query string based on the related type in the summary request.
    
    Args:
        request (SummaryRequest): The summary request containing the related type
        
    Returns:
        str: A query string appropriate for the related type
    """
    if content_type == AIContentType.SHORT_SUMMARY:
        if request.related_type == RelatedType.CHANNEL:
            return "请分析这个频道收录的这些文章的研究主题和科研成果，给首次来到这个频道的读者一个阅读指引"
        elif request.related_type == RelatedType.COLUMN:
            return "请分析这个栏目收录的这些文章的研究主题和科研成果，给首次来到这个栏目的读者一个阅读指引"
        elif request.related_type == RelatedType.ARTICLE:
            return "请分析这个文章的研究主题和科研成果，给首次来到这个文章的读者一个阅读指引"
    elif content_type == AIContentType.QUESTIONS:
        if request.related_type == RelatedType.CHANNEL or request.related_type == RelatedType.COLUMN:
            return "这是一个关于{column_description}的栏目，请根据栏目包含的文献内容提出用户可能会关心的科研问题"
        elif request.related_type == RelatedType.ARTICLE:
            return "这是一个关于{article_description}的文章，请根据文章的摘要提出用户可能会关心的科研问题"
    
    return ""

def _create_params_by_summary_request(request: SummaryRequest) -> dict:
    """
    Create a parameters dictionary based on the summary request.
    
    Args:
        request (SummaryRequest): The summary request containing related type and ID
        
    Returns:
        dict: A dictionary containing the appropriate parameters based on the related type
    """
    if request.related_type == RelatedType.CHANNEL:
        params = {
            "channel_id": request.related_id
        }
    elif request.related_type == RelatedType.COLUMN:
        params = {
            "column_id": request.related_id
        }
    elif request.related_type == RelatedType.ARTICLE:
        params = {
            "article_id": request.related_id
        }
    params["ver"] = request.ver
    params["term_ids"] = request.term_ids
    return params

def _create_query_by_question_request(request: QuestionRequest) -> str:
    """
    Create a query string based on the related type in the question request.
    
    Args:
        request (QuestionRequest): The question request containing the related type
        
    Returns:
        str: A query string appropriate for the related type
    """
    if request.related_type == RelatedType.CHANNEL or request.related_type == RelatedType.COLUMN:
        return "这是一个关于{column_description}的栏目，请根据栏目包含的文献内容提出用户可能会关心的科研问题"
    elif request.related_type == RelatedType.ARTICLE:
        return "这是一个关于{article_description}的文章，请根据文章的摘要提出用户可能会关心的科研问题"
    
    return ""

def _create_params_by_question_request(request: QuestionRequest) -> dict:
    """
    Create a parameters dictionary based on the question request.
    
    Args:
        request (QuestionRequest): The question request containing related type and ID
        
    Returns:
        dict: A dictionary containing the appropriate parameters based on the related type
    """
    if request.related_type == RelatedType.CHANNEL:
        params = {
            "channel_id": request.related_id
        }
    elif request.related_type == RelatedType.COLUMN:
        params = {
            "column_id": request.related_id
        }
    elif request.related_type == RelatedType.ARTICLE:
        params = {
            "article_id": request.related_id
        }
    params["ver"] = request.ver
    params["term_ids"] = request.term_ids
    params["question_count"] = request.count
    return params
    