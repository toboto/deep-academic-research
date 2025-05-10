"""
异步MySQL连接管理模块。

本模块提供异步管理MySQL数据库连接的函数。
"""

import aiomysql
from aiomysql import Pool

# 全局变量，存储活动的数据库连接池
_active_pool = None


async def get_mysql_pool(rbase_db_config: dict) -> Pool:
    """
    获取MySQL数据库连接池，优先复用现有活动连接池

    Args:
        rbase_db_config: 数据库配置字典

    Returns:
        MySQL数据库连接池对象

    Raises:
        ValueError: 如果数据库提供商不是MySQL
        ConnectionError: 如果连接数据库失败
    """
    global _active_pool

    # 检查数据库提供商
    if rbase_db_config.get("provider", "").lower() != "mysql":
        raise ValueError("当前仅支持MySQL数据库")

    # 如果已有活动连接池，尝试复用
    if _active_pool is not None and not _active_pool.closed:
        return _active_pool

    # 创建新的连接池
    try:
        pool = await aiomysql.create_pool(
            host=rbase_db_config.get("config", {}).get("host", "localhost"),
            port=int(rbase_db_config.get("config", {}).get("port", 3306)),
            user=rbase_db_config.get("config", {}).get("username", ""),
            password=rbase_db_config.get("config", {}).get("password", ""),
            db=rbase_db_config.get("config", {}).get("database", ""),
            charset="utf8mb4",
            autocommit=True,
            cursorclass=aiomysql.DictCursor,
            minsize=1,
            maxsize=10,
        )
        _active_pool = pool
        return pool
    except Exception as e:
        raise ConnectionError(f"连接MySQL数据库失败: {e}")


async def close_mysql_pool():
    """
    关闭当前活动的MySQL连接池
    """
    global _active_pool
    if _active_pool is not None and not _active_pool.closed:
        _active_pool.close()
        await _active_pool.wait_closed()
        _active_pool = None 