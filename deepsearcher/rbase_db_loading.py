from typing import List, Tuple

from deepsearcher import configuration
from deepsearcher.rbase.rbase_article import RbaseArticle, RbaseAuthor

# Global variable to store active database connection
_active_connection = None


def init_vector_db(
    collection_name: str, collection_description: str, force_new_collection: bool = False
):
    vector_db = configuration.vector_db

    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    embedding_model = configuration.embedding_model
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )


def _process_authors(
    cursor, article: RbaseArticle, bypass_rbase_db: bool = False
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Process author information, get author IDs and set authors for RbaseArticle object

    Args:
        cursor: Database cursor
        article: RbaseArticle object

    Returns:
        Tuple of author lists and author ID lists
    """
    # Ensure parameters are not None

    if not bypass_rbase_db:
        authors = article.authors or ""
        corresponding_authors = article.corresponding_authors or ""

        author_ids = []
        corresponding_author_ids = []
        # Process regular authors
        author_list = [author.strip() for author in authors.split(",") if author.strip()]

        # Process corresponding authors
        corresponding_author_list = [
            author.strip() for author in corresponding_authors.split(",") if author.strip()
        ]

        # Merge author lists and remove duplicates
        all_authors_set = set(author_list + corresponding_author_list)
        all_authors = list(all_authors_set)

        # Query each author's ID and create RbaseAuthor objects
        for author_name in all_authors:
            # Determine if it's an English name or Chinese name
            is_english = all(ord(c) < 128 for c in author_name)

            # Create author object
            if is_english:
                author_obj = RbaseAuthor(name=author_name, ename=author_name)
                # Query by English name
                author_sql = """
                SELECT id FROM author WHERE ename = %s ORDER BY modified DESC
                """
                cursor.execute(author_sql, (author_name,))
            else:
                author_obj = RbaseAuthor(name=author_name, cname=author_name)
                # Query by Chinese name
                author_sql = """
                SELECT id FROM author WHERE cname = %s ORDER BY modified DESC
                """
                cursor.execute(author_sql, (author_name,))

            # Get all matching author IDs
            author_results = cursor.fetchall()
            if author_results:
                # Extract all author IDs
                ids = [result["id"] for result in author_results]
                # Set author IDs
                author_obj.set_author_ids(ids)
                # Add to ID list
                author_ids.extend(ids)

                # If it's a corresponding author, also add to corresponding author ID list
                is_corresponding = author_name in corresponding_author_list
                author_obj.is_corresponding = is_corresponding
                if is_corresponding:
                    corresponding_author_ids.extend(ids)

            # Add to article
            article.set_author(author_obj)

        return all_authors, author_ids, corresponding_author_list, corresponding_author_ids
