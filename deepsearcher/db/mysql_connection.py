"""
MySQL connection management module.

This module provides functions for managing MySQL database connections.
"""

import pymysql
from pymysql.connections import Connection

# Global variable to store active database connection
_active_connection = None

def get_mysql_connection(rbase_db_config: dict) -> Connection:
    """
    Get MySQL database connection, prioritizing reuse of existing active connection
    
    Args:
        rbase_db_config: Database configuration dictionary
        
    Returns:
        MySQL database connection object
    
    Raises:
        ValueError: If the database provider is not MySQL
        ConnectionError: If connection to database fails
    """
    global _active_connection
    
    # Check database provider
    if rbase_db_config.get('provider', '').lower() != 'mysql':
        raise ValueError("Currently only MySQL database is supported")
    
    # If there is an active connection, try to reuse it
    if _active_connection is not None:
        try:
            # Test if the connection is valid
            _active_connection.ping(reconnect=True)
            return _active_connection
        except Exception:
            # Connection is invalid, close and create a new one
            try:
                _active_connection.close()
            except Exception:
                pass
            _active_connection = None
    
    # Create a new connection
    try:
        conn = pymysql.connect(
            host=rbase_db_config.get('config', {}).get('host', 'localhost'), 
            port=int(rbase_db_config.get('config', {}).get('port', 3306)),
            user=rbase_db_config.get('config', {}).get('username', ''), 
            password=rbase_db_config.get('config', {}).get('password', ''),
            database=rbase_db_config.get('config', {}).get('database', ''), 
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        _active_connection = conn
        return conn
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MySQL database: {e}")

def close_mysql_connection():
    """
    Close the current active MySQL connection
    """
    global _active_connection
    if _active_connection is not None:
        try:
            _active_connection.close()
        except Exception:
            pass
        finally:
            _active_connection = None 