"""
JSON Data Preprocessing Script

This script is used to preprocess data in JSON files, including:
1. Fix garbled Chinese titles (ctitle)
2. Rename fields (pdfFile → pdf_file, txtFile → txt_file)
3. Process author information (authors → author_records, and create new authors field)
4. Process keyword information (keywords → keyword_records, and create new keywords field)
5. Fix garbled Chinese names

Usage:
python process_json_data.py [input_file] [-o OUTPUT] [--sample] [--test-garbled] [--process-garbled]

Arguments:
  input_file         JSON file path to process, must provide this argument or use --sample
  -o, --output      Output file path after processing, defaults to input filename with _processed.json
  --sample          Process small sample file (articles_sample.json) instead of specified file
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from tqdm import tqdm

from deepsearcher import configuration
from deepsearcher.tools.log import color_print, error, info, set_dev_mode, set_level

# Get current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def process_json_data(input_file: str, output_file: str) -> None:
    """
    Process JSON data file

    Args:
        input_file: Input JSON file path
        output_file: Output JSON file path
    """
    # Initialize configuration
    config_path = os.path.join(project_root, "config.rbase.yaml")
    config = configuration.Configuration(config_path)
    configuration.config = config
    configuration.init_config(config)

    # Read JSON file
    info(f"Reading file: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        error(f"JSON parsing error: {str(e)}")
        sys.exit(1)
    except UnicodeDecodeError as e:
        error(f"Encoding error: {str(e)}, please ensure file is UTF-8 encoded")
        sys.exit(1)

    if not isinstance(data, list):
        error("Error: JSON file should contain an array")
        sys.exit(1)

    info(f"Successfully loaded {len(data)} records")

    # Process each record
    total_records = len(data)
    for _, item in tqdm(enumerate(data), total=total_records, desc="Processing records"):
        # 1. Rename fields
        if "pdfFile" in item:
            item["pdf_file"] = item.pop("pdfFile")

        if "txtFile" in item:
            item["txt_file"] = item.pop("txtFile")

        # Ensure article_id field exists
        item["article_id"] = item["id"]

        # 2. Process author information
        if "authors" in item:
            item["author_records"] = item.pop("authors")

            # Process author names
            authors_list = []
            author_ids = []
            corresponding_authors_list = []
            corresponding_author_ids = []

            for author in item["author_records"]:
                # Create new author list
                if "cname" in author and author["cname"]:
                    authors_list.append(author["cname"])
                if "ename" in author and author["ename"]:
                    authors_list.append(author["ename"])

                # Add author ID
                if "id" in author:
                    author_ids.append(author["id"])

                # Identify corresponding authors (relation=2 or 3 indicates corresponding author)
                if "relation" in author and (author["relation"] == 2 or author["relation"] == 3):
                    if "cname" in author and author["cname"]:
                        corresponding_authors_list.append(author["cname"])
                    elif "ename" in author and author["ename"]:
                        corresponding_authors_list.append(author["ename"])

                    if "id" in author:
                        corresponding_author_ids.append(author["id"])

            # Remove duplicates
            item["authors"] = list(set(authors_list))
            item["author_ids"] = list(set(author_ids))

            # Add corresponding author information
            item["corresponding_authors"] = (
                list(set(corresponding_authors_list)) if corresponding_authors_list else []
            )
            item["corresponding_author_ids"] = (
                list(set(corresponding_author_ids)) if corresponding_author_ids else []
            )
        else:
            # If no author information, add empty arrays as default values
            item["author_records"] = []
            item["authors"] = []
            item["author_ids"] = []
            item["corresponding_authors"] = []
            item["corresponding_author_ids"] = []

        # 3. Process keyword information
        if "keywords" in item:
            item["keyword_records"] = item.pop("keywords")

            # Process keywords
            keywords_set = set()
            for keyword in item["keyword_records"]:
                # Add keyword name
                if "name" in keyword:
                    keywords_set.add(keyword["name"])

                # Process concept information
                if "concept" in keyword and keyword["concept"] is not None:
                    concept = keyword["concept"]

                    # Add concept names to keyword set
                    if "cname" in concept and concept["cname"]:
                        keywords_set.add(concept["cname"])
                    if "ename" in concept and concept["ename"]:
                        keywords_set.add(concept["ename"])

            # Save keyword list
            item["keywords"] = list(keywords_set)

        # 4. Process journal information
        if "journalName" in item:
            item["journal_name"] = item.pop("journalName")
        else:
            item["journal_name"] = ""

        if "journalIf" in item:
            item["impact_factor"] = item.pop("journalIf")
        else:
            item["impact_factor"] = 0.0

        # Ensure pubdate field exists
        item["pubdate_str"] = item["pubdate"] if "pubdate" in item else ""
        # Process pubdate field, convert date string to timestamp
        if item["pubdate_str"] and item["pubdate_str"].strip():
            try:
                # Try to parse date string and convert to timestamp
                date_obj = datetime.strptime(item["pubdate_str"], "%Y-%m-%d")
                item["pubdate"] = int(date_obj.timestamp())
            except Exception as e:
                # If parsing fails, keep original pubdate field and log error
                error(f"Date conversion failed: {item['pubdate_str']}, Error: {str(e)}")
                item["pubdate"] = 0  # Set default value to 0
        else:
            # If pubdate_str is empty, set pubdate to 0
            item["pubdate"] = 0

    # Save processed data
    info(f"Saving processed data to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Output statistics
    info(f"Processing completed, processed {total_records} records")

    # Prompt next steps
    color_print("\nNext steps:")
    color_print(
        "1. You can use create_json_vector_db.py script to import the processed data into vector database"
    )
    color_print(
        f"   Example: python examples/create_json_vector_db.py --file {output_file} --collection json_articles"
    )
    color_print(
        f"   \t\t--collection-description 'Academic articles dataset imported from {os.path.basename(input_file)}'"
    )
    color_print(
        "2. Or run without parameters to use default settings: python examples/create_json_vector_db.py"
    )


def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process JSON data file")
    parser.add_argument(
        "input_file",
        nargs="?",
        help="JSON file path to process, must provide this argument or use --sample",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path after processing, defaults to input filename with _processed.json",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process small sample file (articles_sample.json) instead of complete file",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        set_dev_mode(True)
        set_level(logging.DEBUG)
    else:
        set_dev_mode(False)
        set_level(logging.INFO)

    # Determine input file path
    if args.input_file:
        # User provided input file
        if os.path.isabs(args.input_file):
            input_file = args.input_file
        else:
            input_file = os.path.join(os.getcwd(), args.input_file)

        # Default output filename
        input_base = os.path.basename(input_file)
        input_name, input_ext = os.path.splitext(input_base)
        default_output = os.path.join(
            os.path.dirname(input_file), f"{input_name}_processed{input_ext}"
        )

        # Determine output file path
        if args.output:
            if os.path.isabs(args.output):
                output_file = args.output
            else:
                output_file = os.path.join(os.getcwd(), args.output)
        else:
            output_file = default_output

        info(f"Processing file: {input_file}")
        info(f"Output to: {output_file}")
    elif args.sample:
        # Use sample file
        input_file = os.path.join(project_root, "examples", "data", "articles_sample.json")
        output_file = os.path.join(
            project_root, "examples", "data", "articles_sample_processed.json"
        )
        info("Processing sample file...")
    else:
        # Neither input file nor --sample provided, show help and exit
        parser.print_help()
        error("\nError: Must provide input file path or use --sample argument")
        sys.exit(1)

    # Process data
    process_json_data(input_file, output_file)


if __name__ == "__main__":
    main()
