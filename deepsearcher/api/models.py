"""
API数据模型定义

本模块定义了FastAPI接口的请求和响应数据结构。
"""

import hashlib
from enum import Enum
from typing import Optional
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

    def parseFromSummaryRequest(self, request: SummaryRequest):
        # TODO: 尚未完成
        if request.related_type == RelatedType.CHANNEL:
            self.content_type = AIContentType.SHORT_SUMMARY
        elif request.related_type == RelatedType.COLUMN:
            self.content_type = AIContentType.SHORT_SUMMARY
        elif request.related_type == RelatedType.ARTICLE:
            self.content_type = AIContentType.SHORT_SUMMARY
        
        if request.stream:
            self.is_stream_response = StreamResponse.ALLOW
        else:
            self.is_stream_response = StreamResponse.DENY
        self.query = self.createQueryByRequest(request)
        self.params = self.createParamsByRequest(request)
        self.hash()
        self.status = AIRequestStatus.START_REQ
        self.created = datetime.now()
        self.modified = datetime.now()

    
    def createQueryByRequest(self, request: SummaryRequest) -> str:
        if request.related_type == RelatedType.CHANNEL:
            query = f"频道{request.related_id}的概述"
        elif request.related_type == RelatedType.COLUMN:
            query = f"栏目{request.related_id}的概述"
        elif request.related_type == RelatedType.ARTICLE:
            query = f"文章{request.related_id}的概述"
        return query
    
    def createParamsByRequest(self, request: SummaryRequest) -> dict:
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
        return params
    
    def hash(self) -> str:
        self.request_hash = hashlib.md5(self.query.encode()).hexdigest()

class AIResponseStatus(int, Enum):
    """AI响应状态枚举"""
    GENERATING = 1 # 生成中
    FINISH_RES = 10 # 已完成
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
    cache_miss_cnt: int = Field(
        ...,
        description="缓存未命中次数"
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