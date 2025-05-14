"""
API数据模型定义

本模块定义了FastAPI接口的请求和响应数据结构。
"""

from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class ExceptionResponse(BaseModel):
    """异常响应模型"""
    code: int = Field(
        ...,
        description="响应码：0-成功，非0-失败"
    )
    message: str = Field(..., description="响应消息")

class RelatedType(int, Enum):
    """关联类型枚举"""
    CHANNEL = 1  # 频道
    COLUMN = 2   # 栏目
    ARTICLE = 3   # 文章

    @staticmethod
    def IsValid(related_type: int) -> bool:
        return related_type in [RelatedType.CHANNEL, RelatedType.COLUMN, RelatedType.ARTICLE]


class DepressCache(int, Enum):
    """缓存抑制枚举"""
    ENABLE = 1  # 禁用缓存
    DISABLE = 0 # 启用缓存


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
    term_tree_node_ids: Optional[List[int]] = Field(
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
        None,   
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
    term_tree_node_ids: Optional[List[int]] = Field(
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

class DiscussCreateRequest(BaseModel):
    """创建讨论话题请求模型"""
    related_type: RelatedType = Field(
        ...,
        description="关联类型：1-频道，2-栏目，3-文章"
    )
    related_id: Optional[int] = Field(
        None,
        description="关联ID，可选"
    )
    term_tree_node_ids: Optional[List[int]] = Field(
        None,
        description="关键词ID列表，可选"
    )
    ver: int = Field(
        ...,
        description="版本号"
    )
    user_hash: str = Field(
        ...,
        description="用户hash"
    )
    user_id: int = Field(
        ...,
        description="用户ID"
    )



class DiscussCreateResponse(BaseModel):
    """创建讨论话题响应模型"""
    code: int = Field(
        ...,
        description="响应码：0-成功，非0-失败"
    )
    message: str = Field(
        ...,
        description="响应消息"
    )
    thread_uuid: str = Field(
        ...,
        description="话题UUID"
    )

class DiscussPostRequest(BaseModel):
    """发布讨论内容请求模型"""
    thread_uuid: str = Field(
        ...,
        description="话题UUID"
    )
    reply_uuid: str = Field(
        ...,
        description="回复UUID"
    )
    content: str = Field(
        ...,
        description="对话内容"
    )
    user_hash: str = Field(
        ...,
        description="用户hash"
    )
    user_id: int = Field(
        ...,
        description="用户ID"
    )


class DiscussPostResponse(BaseModel):
    """发布讨论内容响应模型"""
    code: int = Field(
        ...,
        description="响应码：0-成功，非0-失败"
    )
    message: str = Field(
        ...,
        description="响应消息"
    )
    uuid: str = Field(
        ...,
        description="讨论UUID"
    )
    depth: int = Field(
        ...,
        description="深度"
    )

class DiscussAIReplyRequest(BaseModel):
    """
    API Request for generating AI reply to discuss
    """
    thread_uuid: str = Field(..., description="UUID of the thread")
    reply_uuid: str = Field(..., description="UUID of the discuss to reply to")
    user_hash: str = Field(..., description="User hash")
    user_id: int = Field(..., description="User ID")


class DiscussListRequest(BaseModel):
    """列出讨论话题请求模型"""
    thread_uuid: str = Field(
        ...,
        description="话题UUID"
    )
    user_hash: Optional[str] = Field(
        "", 
        description="用户hash"
    )
    limit: int = Field(
        ...,
        description="限制数量"
    )
    from_depth: int = Field(
        ...,
        description="从深度开始"
    )

class DiscussListEntity(BaseModel):
    """讨论话题实体"""
    uuid: str = Field(..., description="话题UUID")
    depth: int = Field(..., description="深度")
    content: str = Field(..., description="内容")
    created: datetime = Field(..., description="创建时间")
    role: str = Field(..., description="用户角色")
    user_hash: str = Field(..., description="用户hash")
    user_id: int = Field(..., description="用户ID")
    user_name: str = Field(..., description="用户名")
    user_avatar: str = Field(..., description="用户头像")

class DiscussListResponse(BaseModel):
    """列出讨论话题响应模型"""
    code: int = Field(
        ...,
        description="响应码：0-成功，非0-失败"
    )
    message: str = Field(
        ...,
        description="响应消息"
    )
    count: int = Field(
        ...,
        description="数量"
    )
    discuss_list: Optional[List[DiscussListEntity]] = Field(
        None,
        description="讨论列表"
    )
