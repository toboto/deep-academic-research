"""
Example script for loading article data from JSON file to vector database

This script demonstrates how to extract article data from JSON files and load it into a vector database
for subsequent semantic search and similarity queries.

Usage:
python create_json_vector_db.py [--file FILE_PATH] [--collection COLLECTION_NAME]

Arguments:
  --file FILEPATH         JSON file path to load
  --collection NAME       Collection name in vector database
"""

import json
import os
import sys
import argparse
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.rbase_db_loading import insert_to_vector_db
from deepsearcher.tools.log import info, error, warning

# Suppress unnecessary log output
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def validate_json_file(json_file_path):
    """
    Validate JSON file syntax compliance
    
    Args:
        json_file_path: JSON file path
        
    Returns:
        bool: Whether the file is valid
        str: Error message (if any)
    """
    try:
        # Check if file exists
        if not os.path.exists(json_file_path):
            return False, f"File does not exist: {json_file_path}"
        
        # Try to parse JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        
        return True, "JSON file syntax is correct"
    except json.JSONDecodeError as e:
        return False, f"JSON syntax error: {str(e)}"
    except UnicodeDecodeError as e:
        return False, f"File encoding error: {str(e)}, please ensure file is UTF-8 encoded"
    except Exception as e:
        return False, f"Error during validation: {str(e)}"

class Article:
    """Article data class for converting JSON dictionary to object"""
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)
            
        # Ensure all required fields exist, add default values to avoid AttributeError
        required_fields = [
            "title", "ctitle", "summary", "authors", "author_ids", 
            "corresponding_authors", "corresponding_author_ids", 
            "journal_name", "pubdate", "impact_factor", "keywords", 
            "pdf_file", "txt_file"
        ]
        
        # Set default values
        defaults = {
            "title": "",
            "ctitle": "",
            "summary": "",
            "authors": [],
            "author_ids": [],
            "corresponding_authors": [],
            "corresponding_author_ids": [],
            "journal_name": "",
            "pubdate": "",
            "impact_factor": 0.0,
            "keywords": [],
            "pdf_file": "",
            "txt_file": ""
        }
        
        # Set default values for missing fields
        for field in required_fields:
            if not hasattr(self, field):
                setattr(self, field, defaults.get(field, ""))
    
    def get(self, attr_name, default_value=None):
        """
        Get object attribute value
        
        Args:
            attr_name: Attribute name
            
        Returns:
            Attribute value
            
        Raises:
            AttributeError: When attribute does not exist
        """
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        else:
            if default_value is not None:
                return default_value
            else:
                raise AttributeError(f"Article object has no attribute named '{attr_name}'")

def load_from_json_file(json_file_path):
    """
    Load article data from JSON file
    
    Args:
        json_file_path: JSON file path
        
    Returns:
        list: List of article data objects
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)
    
    if not isinstance(articles_data, list):
        raise ValueError("Data in JSON file is not in array format")
    
    # Convert dictionary data to objects, handle default values
    articles = []
    for article_data in articles_data:
        article = Article(article_data)
        articles.append(article)
    
    info(f"Successfully loaded {len(articles)} article records from JSON file")
    return articles

def main():
    """
    Main function: Load articles from JSON file and store in vector database
    
    Process:
    1. Initialize configuration
    2. Validate and load article data from JSON file
    3. Insert article data into vector database
    """
    # Filter out empty arguments and handle arguments that may contain spaces
    filtered_args = []
    for arg in sys.argv[1:]:
        if not arg.strip():
            continue
        
        # Handle cases like "--file database/json/file.json"
        if arg.startswith('--file '):
            filtered_args.append('--file')
            filtered_args.append(arg[7:].strip())
        elif arg.startswith('--f '):
            filtered_args.append('--file')
            filtered_args.append(arg[3:].strip())
        elif arg.startswith('--collection '):
            filtered_args.append('--collection')
            filtered_args.append(arg[13:].strip())
        elif arg.startswith('-c '):
            filtered_args.append('--collection')
            filtered_args.append(arg[3:].strip())
        elif arg.startswith('--collection-description '):
            filtered_args.append('--collection-description')
            filtered_args.append(arg[25:].strip())
        elif arg.startswith('-d '):
            filtered_args.append('--collection-description')
            filtered_args.append(arg[3:].strip())
        else:
            filtered_args.append(arg)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load article data from JSON file to vector database')
    parser.add_argument('--file', '-f', help='Dataset JSON file path to load')
    parser.add_argument('--collection', '-c', help='Collection name in vector database')
    parser.add_argument('--collection-description', '-d', help='Collection description in vector database')
    
    try:
        args = parser.parse_args(filtered_args)
    except Exception as e:
        error(f"Argument parsing error: {e}")
        info("Running with default arguments...")
        args = argparse.Namespace(file=None, collection=None)
    
    # Step 1: Initialize configuration
    # Get current script directory and build configuration file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, "..", "config.rbase.yaml")
    
    # Load configuration from YAML file
    config = Configuration(yaml_file)

    # Apply configuration to make it globally effective
    init_config(config)

    # Step 2: Validate and load article data from JSON file
    # Set JSON file path
    if args.file:
        json_file_path = args.file
        # If relative path is provided, resolve relative to current directory
        if not os.path.isabs(json_file_path):
            json_file_path = os.path.join(os.getcwd(), json_file_path)
    else:
        # Use sample file by default
        json_file_path = os.path.join(current_dir, "..", "database", "json", "akk_sample_processed.json")
    
    # Validate JSON file
    is_valid, message = validate_json_file(json_file_path)
    if not is_valid:
        error(f"Error: {message}")
        sys.exit(1)
    
    info(f"Validation successful: {message}")
    
    # Load article data from JSON file
    try:
        articles = load_from_json_file(json_file_path)
    except Exception as e:
        error(f"Error loading JSON file: {str(e)}")
        sys.exit(1)
    
    # Step 3: Insert article data into vector database
    # Set vector database collection name and description
    collection_name = args.collection if args.collection else "academic_articles"
    if args.collection_description:
        collection_description = args.collection_description 
    else:
        collection_description = f"Academic articles dataset imported from {os.path.basename(json_file_path)}"

    # Insert article data into vector database
    insert_result = insert_to_vector_db(
        rbase_config=config.rbase_settings,  # Rbase configuration, including database and OSS settings
        articles=articles,                   # List of articles to insert
        collection_name=collection_name,     # Vector database collection name
        collection_description=collection_description,  # Collection description
        force_new_collection=True, # Whether to force create new collection (set to True for first run, can be False later)
        bypass_rbase_db=True
    )
    
    # Print insertion result statistics
    if insert_result:
        # Print number of successfully inserted records
        info(f"Successfully inserted {insert_result.get('insert_count', 0)} records")
        
        # Print ID range of inserted data
        ids = insert_result.get('ids', [])
        if ids:
            min_id = min(ids)
            max_id = max(ids)
            info(f"ID range: {min_id} - {max_id}")
        else:
            warning("No insertion IDs retrieved")

    # Print success message
    info(f"Successfully loaded articles from JSON file into vector database collection '{collection_name}'")


if __name__ == "__main__":
    main() 