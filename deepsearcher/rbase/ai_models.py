import hashlib
import json
import uuid
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from deepsearcher.api.models import SummaryRequest, QuestionRequest, RelatedType, DiscussCreateRequest, DiscussPostRequest

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
    ERROR = 1000 # 错误

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

class DiscussThread(BaseModel):
    """讨论话题模型"""
    id: int = Field(
        ...,
        description="ID"
    )
    uuid: str = Field(
        "",
        description="UUID"
    )
    related_type: RelatedType = Field(
        ...,
        description="关联类型"
    )
    params: dict = Field(
        ...,
        description="参数"
    )
    request_hash: str = Field(
        "",
        description="请求hash"
    )
    user_hash: str = Field(
        "",
        description="用户hash"
    )
    user_id: int = Field(
        0, 
        description="用户ID"
    )
    depth: int = Field(
        0,
        description="深度"
    )
    background: str = Field(
        "",
        description="背景信息"
    )
    is_hidden: int = Field(
        0, 
        description="是否隐藏"
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
        self.request_hash = hashlib.md5(f"{self.related_type}_{json.dumps(self.params, sort_keys=True)}".encode()).hexdigest()

    def create_uuid(self) -> str:
        self.uuid = str(uuid.uuid4())
        return self.uuid

class DiscussRole(str, Enum):
    """讨论角色枚举"""
    USER = "user" # 用户
    ASSISTANT = "assistant" # 助手
    SYSTEM = "system" # 系统

class Discuss(BaseModel):
    """讨论内容模型"""
    id: int = Field(
        0,
        description="ID"
    )
    uuid: str = Field(
        "",
        description="UUID"
    )
    related_type: RelatedType = Field(
        ...,
        description="关联类型"
    )
    thread_id: int = Field(
        None,
        description="话题ID"
    )
    thread_uuid: str = Field(
        ...,
        description="话题UUID"
    )
    reply_id: Optional[int] = Field(
        None,
        description="回复ID"
    )
    reply_uuid: Optional[str] = Field(
        None,
        description="回复UUID"
    )
    depth: int = Field(
        0,
        description="深度"
    )
    content: str = Field(
        "",
        description="内容"
    )
    tokens: dict = Field(
        ...,
        description="正在生成的tokens"
    )
    usage: dict = Field(
        ...,
        description="使用情况"
    )
    role: DiscussRole = Field(
        ...,
        description="角色"
    )
    user_id: Optional[int] = Field(
        None,
        description="用户ID"
    )
    is_hidden: int = Field(
        0,
        description="是否隐藏"
    )
    like: int = Field(
        0,
        description="点赞数"
    )
    trample: int = Field(
        0,
        description="踩数"
    )
    is_summary: int = Field(
        0,
        description="是否存在总结"
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
    
    def create_uuid(self) -> str:
        self.uuid = str(uuid.uuid4())
        return self.uuid


class TermTreeNode(BaseModel):
    """术语树节点模型"""
    id: int = Field(
        ...,
        description="ID"
    )
    tree_id: int = Field(
        ...,
        description="树ID"
    )
    parent_node_id: int = Field(
        ...,
        description="父节点ID"
    )
    node_concept_name: str = Field(
        ...,
        description="节点概念名称"
    )
    node_concept_id: int = Field(
        ...,
        description="节点概念ID"
    )
    intro: Optional[str] = Field(
        None,
        description="介绍"
    )
    sequence: Optional[int] = Field(
        0,
        description="顺序"
    )
    children_count: Optional[int] = Field(
        0,
        description="子节点数量"
    )
    status: int = Field(
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

class Base(BaseModel):
    """用户库模型"""
    id: int = Field(
        ...,
        description="ID"
    )
    uuid: str = Field(
        ...,
        description="UUID"
    )
    name: str = Field(
        ...,
        description="名称"
    )
    intro: Optional[str] = Field(
        None,
        description="介绍"
    )
    created: Optional[datetime] = Field(
        None,
        description="创建时间"
    )
    modified: Optional[datetime] = Field(
        None,
        description="更新时间"
    )

class BaseCategory(BaseModel):
    """用户库分类模型"""
    id: int = Field(
        ...,
        description="ID"
    )
    alias: str = Field(
        ...,
        description="别名"
    )
    base_id: int = Field(
        ...,
        description="基础ID"
    )
    type: int = Field(
        ...,
        description="类型"
    )
    name: str = Field(
        ...,
        description="名称"
    )
    base_name: Optional[str] = Field(
        None,
        description="用户库名称"
    )
    status: int = Field(
        ...,
        description="状态"
    )
    created: Optional[datetime] = Field(
        None,
        description="创建时间"
    )
    modified: Optional[datetime] = Field(
        None,
        description="更新时间"
    )

def initialize_ai_request_by_summary(request: SummaryRequest, metadata: dict = {}):
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
        query=_create_query_by_summary_request(request, AIContentType.SHORT_SUMMARY, metadata),
        params=_create_params_by_summary_request(request, metadata),
        request_hash="",
        status=AIRequestStatus.START_REQ,
        created=datetime.now(),
        modified=datetime.now()
    )
    ai_request.hash()
    return ai_request

def initialize_ai_request_by_question(request: QuestionRequest, metadata: dict = {}):
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
        query=_create_query_by_question_request(request, metadata),
        params=_create_params_by_question_request(request, metadata),
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

def _create_query_by_summary_request(request: SummaryRequest, content_type: AIContentType, metadata: dict = {}) -> str:
    """
    Create a query string based on the related type in the summary request.
    
    Args:
        request (SummaryRequest): The summary request containing the related type
        
    Returns:
        str: A query string appropriate for the related type
    """
    if request.related_type == RelatedType.CHANNEL or request.related_type == RelatedType.COLUMN:
        if metadata.get('column_description'):
            return f"这是一个{metadata.get('column_description')}，请分析这个栏目收录的这些文章的研究主题和科研成果，给首次来到这个栏目的读者一个阅读指引"
        else:
            return "请分析这个栏目收录的这些文章的研究主题和科研成果，给首次来到这个栏目的读者一个阅读指引"
    elif request.related_type == RelatedType.ARTICLE:
        if metadata.get('article_title'):
            query = f"这篇文章标题是：{metadata.get('article_title')}"
            if metadata.get('article_abstract'):
                query += f"\n摘要：{metadata.get('article_abstract')}\n"
            query += "请分析这个文章的研究主题和科研成果，给首次来到这个文章的读者一个阅读指引"
            return query
        else:
            return "请分析这个文章的研究主题和科研成果，给首次来到这个文章的读者一个阅读指引"
    
    return ""

def _create_params_by_summary_request(request: SummaryRequest, metadata: dict = {}) -> dict:
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
        if metadata.get('base_id'):
            params["channel_id"] = metadata.get('base_id')
    elif request.related_type == RelatedType.ARTICLE:
        params = {
            "article_id": request.related_id
        }
    params["ver"] = request.ver
    params["term_tree_node_ids"] = request.term_tree_node_ids
    return params

def _create_query_by_question_request(request: QuestionRequest, metadata: dict = {}) -> str:
    """
    Create a query string based on the related type in the question request.
    
    Args:
        request (QuestionRequest): The question request containing the related type
        
    Returns:
        str: A query string appropriate for the related type
    """
    if request.related_type == RelatedType.CHANNEL or request.related_type == RelatedType.COLUMN:
        if metadata.get('column_description'):
            return f"这是一个{metadata.get('column_description')}，请根据栏目包含的文献内容提出用户可能会关心的科研问题"
        else:
            return "请根据栏目包含的文献内容提出用户可能会关心的科研问题"
    elif request.related_type == RelatedType.ARTICLE:
        if metadata.get('article_title'):
            query = f"这篇文章标题是：{metadata.get('article_title')}"
            if metadata.get('article_abstract'):
                query += f"\n摘要：{metadata.get('article_abstract')}\n"
            query += "\n请根据文章的摘要提出用户可能会关心的科研问题"
            return query
        else:
            return "请根据文章的摘要提出用户可能会关心的科研问题"
    
    return ""

def _create_params_by_question_request(request: QuestionRequest, metadata: dict = {}) -> dict:
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
        if metadata.get('base_id'):
            params["channel_id"] = metadata.get('base_id')
    elif request.related_type == RelatedType.ARTICLE:
        params = {
            "article_id": request.related_id
        }
    params["ver"] = request.ver
    params["term_tree_node_ids"] = request.term_tree_node_ids
    params["question_count"] = request.count
    return params

def initialize_discuss_thread(request: DiscussCreateRequest) -> DiscussThread:
    """
    Initialize a DiscussThread object from a DiscussCreateRequest.
    
    Args:
        request (DiscussCreateRequest): The source discuss create request
        
    Returns:
        DiscussThread: A new discuss thread object initialized with values from the discuss create request
    """
    params = {}
    if request.related_type == RelatedType.CHANNEL:
        params["channel_id"] = request.related_id
    elif request.related_type == RelatedType.COLUMN:
        params["column_id"] = request.related_id
    elif request.related_type == RelatedType.ARTICLE:
        params["article_id"] = request.related_id

    if request.term_tree_node_ids:
        params["term_tree_node_ids"] = request.term_tree_node_ids
    if request.ver is not None:
        params["ver"] = request.ver

    discuss_thread = DiscussThread(
        id=0,
        uuid="",
        related_type=request.related_type,
        params=params,
        request_hash="",
        user_hash=request.user_hash,
        user_id=request.user_id,
        created=datetime.now(),
        modified=datetime.now()
    )
    discuss_thread.hash()
    discuss_thread.create_uuid()
    return discuss_thread

def initialize_discuss(request: DiscussPostRequest, thread: DiscussThread, reply_id: int = 0) -> Discuss:
    """
    从DiscussPostRequest初始化Discuss对象
    
    Args:
        request (DiscussPostRequest): 讨论内容请求
        
    Returns:
        DiscussContent: 新的讨论内容对象
    """
    discuss = Discuss(
        id=0,
        thread_id=thread.id,
        thread_uuid=thread.uuid,
        reply_id=reply_id,
        reply_uuid=request.reply_uuid,
        content=request.content,
        user_hash=request.user_hash,
        user_id=request.user_id,
        created=datetime.now(),
        modified=datetime.now()
    )
    discuss.create_uuid()
    return discuss

