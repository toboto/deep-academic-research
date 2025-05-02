# API 文档

本文档详细说明了 Deep Academic Research 系统提供的 RESTful API 接口。

## 基础信息

- 基础URL: `http://localhost:8000/api/v1`
- 所有请求和响应均使用 JSON 格式
- 认证方式：API Key（在请求头中添加 `X-API-Key`）

## 接口列表

### 1. 生成研究主题综述

#### 请求

```http
POST /overview
```

#### 请求头

```http
Content-Type: application/json
X-API-Key: your-api-key
```

#### 请求体

```json
{
    "topic": "人工智能在医疗领域的应用",
    "language": "zh",  // 可选，默认 "zh"，支持 "zh" 或 "en"
    "top_k_per_section": 20,  // 可选，每个章节检索的文档数量
    "top_k_accepted_results": 20,  // 可选，每个章节接受的文档数量
    "vector_db_collection": "default"  // 可选，向量数据库集合名称
}
```

#### 响应

```json
{
    "status": "success",
    "data": {
        "english_response": "# Overview: AI Applications in Healthcare\n\n## Abstract\n...",
        "chinese_response": "# 综述：人工智能在医疗领域的应用\n\n## 摘要\n..."
    },
    "tokens_used": 12345
}
```

### 2. 生成研究者成果综述

#### 请求

```http
POST /personal
```

#### 请求头

```http
Content-Type: application/json
X-API-Key: your-api-key
```

#### 请求体

```json
{
    "researcher": "张三",
    "language": "zh",  // 可选，默认 "zh"，支持 "zh" 或 "en"
    "top_k_per_section": 20,  // 可选，每个章节检索的文档数量
    "top_k_accepted_results": 20,  // 可选，每个章节接受的文档数量
    "vector_db_collection": "default"  // 可选，向量数据库集合名称
}
```

#### 响应

```json
{
    "status": "success",
    "data": {
        "english_response": "# Research Overview: Zhang San\n\n## Academic Background\n...",
        "chinese_response": "# 科研综述：张三\n\n## 学术背景\n..."
    },
    "tokens_used": 12345
}
```

### 3. 健康检查

#### 请求

```http
GET /health
```

#### 请求头

```http
X-API-Key: your-api-key
```

#### 响应

```json
{
    "status": "success",
    "data": {
        "version": "1.0.0",
        "services": {
            "llm": "healthy",
            "vector_db": "healthy",
            "translator": "healthy"
        }
    }
}
```

## 错误处理

所有接口在发生错误时会返回相应的 HTTP 状态码和错误信息：

```json
{
    "status": "error",
    "error": {
        "code": "ERROR_CODE",
        "message": "错误描述信息"
    }
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

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000/api/v1"

# 生成研究主题综述
response = requests.post(
    f"{BASE_URL}/overview",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    },
    json={
        "topic": "人工智能在医疗领域的应用",
        "language": "zh"
    }
)

if response.status_code == 200:
    result = response.json()
    print(result["data"]["chinese_response"])
else:
    print(f"Error: {response.json()['error']['message']}")
```

### cURL 示例

```bash
# 生成研究主题综述
curl -X POST \
  http://localhost:8000/api/v1/overview \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-api-key' \
  -d '{
    "topic": "人工智能在医疗领域的应用",
    "language": "zh"
  }'
```

## 注意事项

1. API 调用频率限制：
   - 每个 API Key 每分钟最多 60 次请求
   - 每个 IP 每分钟最多 100 次请求

2. 响应时间：
   - 健康检查接口：< 1s
   - 综述生成接口：30s - 5min（取决于主题复杂度和数据量）

3. 数据安全：
   - 所有请求和响应都通过 HTTPS 加密
   - API Key 请妥善保管，不要泄露
   - 建议定期更换 API Key

4. 错误处理：
   - 建议实现重试机制
   - 对于长时间运行的请求，建议实现超时处理
   - 建议实现错误日志记录 