# API 文档

本文档详细说明了 Deep Academic Research 系统提供的 RESTful API 接口。

## 基础信息

- 基础URL: `/api`
- 所有请求和响应均使用 JSON 格式

## 接口列表

### 1. 生成 AI 概述

#### 请求

```http
POST /generate/summary
```

#### 请求头

```http
Content-Type: application/json
```

#### 请求体

```json
{
    "related_type": 1,  // 可选值: 1-频道，2-栏目，3-文章
    "related_id": 123,  // 关联ID
    "term_tree_node_ids": [1, 2, 3],  // 可选，术语树节点ID列表
    "ver": 1,  // 可选，版本号
    "depress_cache": 1,  // 可选，缓存抑制：0-启用缓存，1-禁用缓存
    "stream": true,  // 可选，是否使用流式响应
    "discuss_thread_uuid": "uuid",  // 可选，讨论主题UUID
    "discuss_reply_uuid": "uuid"  // 可选，回复UUID
}
```

#### 响应

```json
{
    "code": 0,
    "message": "success",
    "content": "生成的概述内容..."
}
```

### 2. 生成推荐问题

#### 请求

```http
POST /generate/questions
```

#### 请求头

```http
Content-Type: application/json
```

#### 请求体

```json
{
    "related_type": 1,  // 可选值: 1-频道，2-栏目，3-文章
    "related_id": 123,  // 关联ID
    "term_tree_node_ids": [1, 2, 3],  // 可选，术语树节点ID列表
    "ver": 1,  // 可选，版本号
    "depress_cache": 1,  // 可选，缓存抑制：0-启用缓存，1-禁用缓存
    "count": 3  // 问题数量
}
```

#### 响应

```json
{
    "code": 0,
    "message": "success",
    "questions": ["问题1", "问题2", "问题3"]
}
```

### 3. 创建讨论主题

#### 请求

```http
POST /generate/discuss_create
```

#### 请求头

```http
Content-Type: application/json
```

#### 请求体

```json
{
    "related_type": 1,  // 可选值: 1-频道，2-栏目，3-文章
    "related_id": 123,  // 关联ID
    "term_tree_node_ids": [1, 2, 3],  // 可选，术语树节点ID列表
    "ver": 1,  // 可选，版本号
    "user_hash": "user_hash",  // 用户哈希值
    "user_id": 123  // 用户ID
}
```

#### 响应

```json
{
    "code": 0,
    "message": "success",
    "thread_uuid": "uuid",
    "depth": 1,
    "has_summary": false
}
```

### 4. 获取讨论列表

#### 请求

```http
GET /generate/list_discuss
```

#### 请求头

```http
```

#### 查询参数

```json
{
    "thread_uuid": "uuid",  // 讨论主题UUID
    "user_hash": "user_hash",  // 用户哈希值
    "from_depth": 0,  // 起始深度
    "limit": 20,  // 返回数量限制
    "sort": 1  // 排序方式：1-升序，-1-降序
}
```

#### 响应

```json
{
    "code": 0,
    "message": "success",
    "count": 10,
    "discuss_list": [
        {
            "uuid": "uuid",
            "depth": 1,
            "content": "讨论内容",
            "created": 1234567890,
            "role": "user",
            "is_summary": false,
            "user_hash": "user_hash",
            "user_id": 123,
            "user_name": "用户名",
            "user_avatar": "头像URL"
        }
    ]
}
```

### 5. 发布讨论内容

#### 请求

```http
POST /generate/discuss_post
```

#### 请求头

```http
Content-Type: application/json
```

#### 请求体

```json
{
    "thread_uuid": "uuid",  // 讨论主题UUID
    "reply_uuid": "uuid",  // 可选，回复的UUID
    "content": "讨论内容",
    "user_hash": "user_hash",  // 用户哈希值
    "user_id": 123  // 可选，用户ID
}
```

#### 响应

```json
{
    "code": 0,
    "message": "success",
    "uuid": "uuid",
    "depth": 1
}
```

### 6. AI 回复讨论

#### 请求

```http
POST /generate/ai_reply
```

#### 请求头

```http
Content-Type: application/json
```

#### 请求体

```json
{
    "thread_uuid": "uuid",  // 讨论主题UUID
    "reply_uuid": "uuid"  // 回复的UUID
}
```

#### 响应

流式响应，格式如下：

```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "rbase-discuss-agent",
    "choices": [{
        "index": 0,
        "delta": {"content": "回复内容片段"},
        "finish_reason": null
    }]
}
```

## 错误处理

所有接口在发生错误时会返回相应的 HTTP 状态码和错误信息：

```json
{
    "code": 500,
    "message": "错误描述信息"
}
```

常见错误码：

- `400`: 请求参数错误
- `401`: 未授权（API Key 无效）
- `403`: 权限不足
- `404`: 资源不存在
- `500`: 服务器内部错误

## 使用示例

### Python 示例

```python
import requests

BASE_URL = "https://yourdomain.com/api"

# 生成 AI 概述
response = requests.post(
    f"{BASE_URL}/generate/summary",
    headers={
        "Content-Type": "application/json"
    },
    json={
        "related_type": 1,
        "related_id": 123,
        "term_tree_node_ids": [1, 2, 3],
        "stream": True,
        "depress_cache": 1
    }
)

if response.status_code == 200:
    result = response.json()
    print(result["content"])
else:
    print(f"Error: {response.json()['message']}")
```

### cURL 示例

```bash
# 生成 AI 概述
curl -X POST \
  https://yourdomain.com/api/generate/summary \
  -H 'Content-Type: application/json' \
  -d '{
    "related_type": 1,
    "related_id": 123,
    "term_tree_node_ids": [1, 2, 3],
    "stream": true,
    "depress_cache": 1
  }'
```

## 注意事项

1. API 调用频率限制：
   - 每个 API Key 每分钟最多 60 次请求
   - 每个 IP 每分钟最多 100 次请求

2. 响应时间：
   - 普通请求：< 1s
   - AI 生成请求：30s - 5min（取决于内容复杂度和数据量）

3. 数据安全：
   - 所有请求和响应都通过 HTTPS 加密
   - API Key 请妥善保管，不要泄露
   - 建议定期更换 API Key

4. 错误处理：
   - 建议实现重试机制
   - 对于长时间运行的请求，建议实现超时处理
   - 建议实现错误日志记录 