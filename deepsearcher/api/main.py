"""
FastAPI Application Entry Point

This module initializes and configures the FastAPI application.
"""

import os
import logging
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from deepsearcher import configuration
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.api.routes import router
from deepsearcher.api.models import ExceptionResponse
from deepsearcher.tools.log import set_dev_mode, set_level

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    CONFIG_FILE_PATH: str = "yaml"

    model_config = SettingsConfigDict(env_file=".env")

# Configure logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    s = Settings()
    if os.path.exists(s.CONFIG_FILE_PATH):
        config_file = s.CONFIG_FILE_PATH
    else:
        config_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "..",
            s.CONFIG_FILE_PATH
        )

    logger.info("Initializing Rbase API...")
    setattr(configuration, 'config', Configuration(config_file))
    # 初始化配置
    init_config(configuration.config)
    
    # 配置日志
    rbase_settings = configuration.config.rbase_settings
    api_settings = rbase_settings.get('api', {})
    log_file = api_settings.get('log_file', 'logs/api.log')
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Rbase API initialized successfully")
    yield
    logger.info("Shutting down Rbase API...")

# Initialize FastAPI app
app = FastAPI(
    title="Rbase Deep Searcher API",
    description="API for Rbase academic research platform",
    version="0.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# Include routers
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    """
    根路径处理函数，返回API基本信息
    """
    return {
        "name": "Rbase Deep Searcher API",
        "version": "0.0.1",
        "description": "Rbase Deep Searcher API服务，提供AI概述和推荐问题功能",
    }

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy"}

# 获取服务器配置从配置文件
def get_server_config(config_path: str = "config.rbase.yaml"):
    """
    从配置文件中获取服务器配置
    
    Returns:
        tuple: (host, port) 主机地址和端口
    """
    try:
        conf = Configuration(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "..",
                config_path
            )
        )
        rbase_settings = conf.rbase_settings
        api_settings = rbase_settings.get('api', {})
        host = api_settings.get('host', '0.0.0.0')
        port = int(api_settings.get('port', 8000))
        return host, port
    except Exception as e:
        logger.error(f"获取服务器配置失败: {e}")
        return '0.0.0.0', 8000

# 添加全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """格式化请求验证错误信息"""
    error_messages = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = f"字段 '{field}' 验证失败: {error['msg']}"
        error_messages.append(message)
    
    return JSONResponse(
        status_code=400,
        content=ExceptionResponse(
            code=400, 
            message="请求参数验证失败:\n" + "\n".join(error_messages)
        ).model_dump()
    )

# 添加其他异常处理器
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ExceptionResponse(code=500, message=str(exc)).model_dump()
    )

# 当作为主程序运行时，使用配置文件中的设置启动服务器
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rbase API")
    parser.add_argument("--verbose", "-v", action="store_true", help="是否开启详细日志")
    args = parser.parse_args()

    host, port = get_server_config(args.config)
    logger.info(f"Starting server at {host}:{port}")

    # 读取日志文件配置
    rbase_settings = configuration.config.rbase_settings
    api_settings = rbase_settings.get('api', {})
    log_file = api_settings.get('log_file', 'logs/api.log')
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置uvicorn的日志
    import uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    
    # 统一日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = log_format
    log_config["formatters"]["access"]["fmt"] = log_format
    
    # 添加文件处理器到所有日志配置
    for logger_name in log_config["loggers"]:
        logger_conf = log_config["loggers"][logger_name]
        if args.verbose:
            set_dev_mode(True)
            set_level(logging.DEBUG)
            logger_conf["level"] = "DEBUG"
            logger_conf["handlers"] = ["default", "file"]
        else:
            set_dev_mode(False)
            set_level(logging.INFO)
            logger_conf["level"] = "INFO"
            logger_conf["handlers"] = ["file"]
    
    # 定义文件处理器
    log_config["handlers"]["file"] = {
        "class": "logging.FileHandler",
        "formatter": "default",
        "filename": log_file
    }
    
    uvicorn_config = uvicorn.Config("deepsearcher.api.main:app", host=host, port=port, reload=True, log_config=log_config)
    server = uvicorn.Server(uvicorn_config)
    server.run()