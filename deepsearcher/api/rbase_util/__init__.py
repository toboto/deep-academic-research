"""
Rbase Database Utilities

This package contains database utility functions for Rbase.
"""

from .ai_content import (
    get_response_by_request_hash,
    save_request_to_db,
    save_response_to_db,
)
from .discuss import (
    update_ai_content_to_discuss,
    update_discuss_thread_depth,
    get_discuss_thread_by_request_hash,
    is_thread_has_summary,
    get_discuss_thread_by_id,
    save_discuss_thread,
    get_discuss_thread_by_uuid,
    get_discuss_by_uuid,
    save_discuss,
    get_discuss_in_thread,
    get_discuss_thread_history,
    list_discuss_in_thread,
)
from .metadata import (
    get_term_tree_nodes,
    get_base_by_id,
    get_base_category_by_id,
)
from .utils import get_request_hash

__all__ = [
    # AI Content
    'get_response_by_request_hash',
    'save_request_to_db',
    'save_response_to_db',
    
    # Discuss
    'update_ai_content_to_discuss',
    'update_discuss_thread_depth',
    'get_discuss_thread_by_request_hash',
    'is_thread_has_summary',
    'get_discuss_thread_by_id',
    'save_discuss_thread',
    'get_discuss_thread_by_uuid',
    'get_discuss_by_uuid',
    'save_discuss',
    'get_discuss_in_thread',
    'get_discuss_thread_history',
    'list_discuss_in_thread',
    
    # Metadata
    'get_term_tree_nodes',
    'get_base_by_id',
    'get_base_category_by_id',
    
    # Utils
    'get_request_hash',
] 