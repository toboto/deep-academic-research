"""
Metadata Database Operations

This module contains database operations for metadata.
"""

from deepsearcher import configuration
from deepsearcher.db.async_mysql_connection import get_mysql_pool
from deepsearcher.rbase.ai_models import (
    TermTreeNode,
    Base,
    BaseCategory,
)

async def get_term_tree_nodes(term_tree_node_ids: list[int]) -> list[TermTreeNode]:
    """
    Get term tree nodes
    """
    if not term_tree_node_ids:
        return []

    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                placeholders = ', '.join(['%s'] * len(term_tree_node_ids))
                sql = """
                SELECT * FROM term_tree_node WHERE id IN ({placeholders})
                """
                sql = sql.format(placeholders=placeholders)
                await cursor.execute(sql, term_tree_node_ids)
                results = await cursor.fetchall()
                return [TermTreeNode(**result) for result in results]
    except Exception as e:
        raise Exception(f"Failed to get term tree nodes: {e}")

async def get_base_by_id(id: int) -> Base:
    """
    Get base by ID
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                sql = "SELECT * FROM base WHERE id = %s"
                await cursor.execute(sql, (id,))
                result = await cursor.fetchone()
                return Base(**result)
    except Exception as e:
        raise Exception(f"Failed to get base by id: {e}")

async def get_base_category_by_id(id: int, base_id: int = 0) -> BaseCategory:
    """
    Get base category by ID
    """
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                if base_id:
                    sql = """SELECT bc.*, b.name as base_name FROM base_category bc 
                             LEFT JOIN base b ON bc.base_id=b.id WHERE bc.id=%s AND bc.base_id=%s"""
                    await cursor.execute(sql, (id, base_id))
                else:
                    sql = """SELECT bc.*, b.name as base_name FROM base_category bc 
                             LEFT JOIN base b ON bc.base_id=b.id WHERE bc.id = %s"""
                    await cursor.execute(sql, (id,))
                result = await cursor.fetchone()
                return BaseCategory(**result)
    except Exception as e:
        raise Exception(f"Failed to get base category by id: {e}") 