"""
Utility Functions

This module contains utility functions for database operations.
"""

import json
import hashlib
from pydantic import BaseModel

def get_request_hash(request: BaseModel) -> str:
    """
    Calculate request hash value
    
    Args:
        request: Request object
        
    Returns:
        str: Request hash value
    """
    # Convert request object to dictionary
    request_dict = request.model_dump()
    
    # Remove fields that should not participate in hash calculation
    if 'user_hash' in request_dict:
        del request_dict['user_hash']
    if 'user_id' in request_dict:
        del request_dict['user_id']
        
    # Convert dictionary to JSON string
    request_json = json.dumps(request_dict, sort_keys=True)
    
    # Calculate hash value
    return hashlib.md5(request_json.encode()).hexdigest() 