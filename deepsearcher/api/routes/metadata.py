"""
Metadata Builder

This module contains functions for building metadata for AI requests.
"""

from typing import List
from deepsearcher.api.models import RelatedType
from deepsearcher.rbase.ai_models import DiscussRole
from deepsearcher.api.rbase_util import (
    get_term_tree_nodes,
    get_base_by_id,
    get_base_category_by_id,
    get_discuss_thread_by_uuid,
    get_discuss_thread_history,
)
from deepsearcher.rbase_db_loading import load_articles_by_article_ids

async def build_metadata(related_type: RelatedType, related_id: int, term_tree_node_ids: List[int] = []) -> dict:
    """
    Build metadata for general AI requests
    """
    metadata = await build_metadata_by_term_tree_node(term_tree_node_ids)
    metadata = await build_metadata_by_related_type(related_type, related_id, metadata)
    return metadata

async def build_metadata_by_term_tree_node(term_tree_node_ids: List[int]) -> dict:
    """
    Build metadata for term tree nodes.
    """
    term_tree_nodes = await get_term_tree_nodes(term_tree_node_ids)
    metadata = {"concepts": []}
    for node in term_tree_nodes:
        metadata["concepts"].append(node.node_concept_name)
    metadata["column_description"] = "、".join(metadata["concepts"])
    return metadata

async def build_metadata_by_related_type(related_type: RelatedType, related_id: int, metadata: dict = {}) -> dict:
    """
    Build metadata for related type.
    """
    desc = metadata.get("column_description", "")
    if related_type == RelatedType.CHANNEL:
        base = await get_base_by_id(related_id)
        if base:
            metadata["base_id"] = base.id
            metadata["column_description"] = f"频道：{base.name} 中，内容关于：{desc} 的栏目"
    elif related_type == RelatedType.COLUMN:
        base_category = await get_base_category_by_id(related_id)
        if base_category:
            metadata["base_id"] = base_category.base_id
            metadata["column_description"] = f"栏目：{base_category.name}, 内容关于：{desc}"
    elif related_type == RelatedType.ARTICLE:
        metadata = await build_article_metadata(related_id, metadata)
    return metadata

async def build_article_metadata(article_id: int, metadata: dict = {}) -> dict:
    """
    Build metadata for article.
    """
    articles = await load_articles_by_article_ids([article_id])
    if len(articles) > 0:
        metadata["article_id"] = articles[0].article_id
        metadata["article_title"] = articles[0].title
        metadata["article_abstract"] = articles[0].abstract
    return metadata 

async def build_metadata_by_discuss_thread(thread_uuid: str, metadata: dict = {}) -> dict:
    """
    Build metadata for discuss thread.
    """
    if not thread_uuid:
        return metadata
    thread = await get_discuss_thread_by_uuid(thread_uuid)
    if thread and thread.depth > 0:
        discuss_history = await get_discuss_thread_history(thread.id, 0, 3, role=DiscussRole.USER)
        metadata["user_history"] = discuss_history
    return metadata