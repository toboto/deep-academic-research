"""
FastAPI应用主文件

本模块定义了FastAPI应用的主入口，包括应用配置和路由注册。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deepsearcher.api.routes import router

# 创建FastAPI应用
app = FastAPI(
    title="Rbase API",
    description="Rbase API服务，提供AI概述和推荐问题功能",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 注册路由
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """
    根路径处理函数，返回API基本信息
    """
    return {
        "name": "Rbase API",
        "version": "0.1.0",
        "description": "Rbase API服务，提供AI概述和推荐问题功能",
    }


@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"} 