from typing import List, Optional, Union

import numpy as np
from pymilvus import DataType, MilvusClient

from deepsearcher.loader.splitter import Chunk
from deepsearcher.tools import log
from deepsearcher.vector_db.base import BaseVectorDB, CollectionInfo, RetrievalResult


class Milvus(BaseVectorDB):
    """Milvus vector database implementation that extends BaseVectorDB."""

    client: MilvusClient = None

    def __init__(
        self,
        default_collection: str = "deepsearcher",
        uri: str = "http://localhost:19530",
        token: str = "root:Milvus",
        db: str = "default",
    ):
        """
        Initialize Milvus client with connection parameters.

        Args:
            default_collection: Name of the default collection
            uri: Milvus server URI
            token: Authentication token
            db: Database name
        """
        super().__init__(default_collection)
        self.default_collection = default_collection
        self.client = MilvusClient(uri=uri, token=token, db_name=db, timeout=30)

    def init_collection(
        self,
        dim: int,
        collection: Optional[str] = "deepsearcher",
        description: Optional[str] = "",
        force_new_collection: bool = False,
        text_max_length: int = 65_535,
        reference_max_length: int = 2048,
        metric_type: str = "L2",
        *args,
        **kwargs,
    ):
        """
        Initialize a new collection with specified schema and indexes.

        Args:
            dim: Dimension of the vector embeddings
            collection: Collection name
            description: Collection description
            force_new_collection: Whether to drop existing collection
            text_max_length: Maximum length for text field
            reference_max_length: Maximum length for reference field
            metric_type: Distance metric type for vector similarity
        """
        if not collection:
            collection = self.default_collection
        if description is None:
            description = ""
        try:
            has_collection = self.client.has_collection(collection, timeout=5)
            if force_new_collection and has_collection:
                self.client.drop_collection(collection)
            elif has_collection:
                return
            schema = self.client.create_schema(
                enable_dynamic_field=False, auto_id=True, description=description
            )
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("text", DataType.VARCHAR, max_length=text_max_length)
            schema.add_field("reference", DataType.VARCHAR, max_length=reference_max_length)
            schema.add_field("reference_id", DataType.INT64)
            schema.add_field(
                "keywords",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=500,
                max_length=200,
            )
            schema.add_field(
                "authors",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=200,
                max_length=100,
            )
            schema.add_field(
                "author_ids", DataType.ARRAY, element_type=DataType.INT64, max_capacity=500
            )
            schema.add_field(
                "corresponding_authors",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=40,
                max_length=100,
            )
            schema.add_field(
                "corresponding_author_ids",
                DataType.ARRAY,
                element_type=DataType.INT64,
                max_capacity=100,
            )
            schema.add_field("base_ids", DataType.ARRAY, element_type=DataType.INT64, max_capacity=100)
            schema.add_field("impact_factor", DataType.FLOAT)
            schema.add_field("rbase_factor", DataType.FLOAT)
            schema.add_field("pubdate", DataType.INT64)
            schema.add_field("metadata", DataType.JSON, nullable=True)
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="embedding", metric_type=metric_type)
            index_params.add_index(field_name="keywords", index_type="", index_name="keywords_idx")
            index_params.add_index(field_name="authors", index_type="", index_name="authors_idx")
            index_params.add_index(
                field_name="author_ids", index_type="", index_name="author_ids_idx"
            )
            index_params.add_index(
                field_name="corresponding_authors",
                index_type="",
                index_name="corresponding_authors_idx",
            )
            index_params.add_index(
                field_name="corresponding_author_ids",
                index_type="",
                index_name="corresponding_author_ids_idx",
            )
            index_params.add_index(
                field_name="base_ids", index_type="", index_name="base_ids_idx"
            )
            index_params.add_index(
                field_name="impact_factor", index_type="", index_name="impact_factor_idx"
            )
            index_params.add_index(
                field_name="rbase_factor", index_type="", index_name="rbase_factor_idx"
            )
            index_params.add_index(field_name="pubdate", index_type="", index_name="pubdate_idx")
            self.client.create_collection(
                collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )
            log.color_print(f"create collection [{collection}] successfully")
        except Exception as e:
            log.critical(f"fail to init db for milvus, error info: {e}")

    def insert_data(
        self,
        collection: Optional[str],
        chunks: List[Chunk],
        batch_size: int = 256,
        *args,
        **kwargs,
    ):
        """
        Insert data into the vector database.

        Args:
            collection: Collection name
            chunks: List of data chunks to insert
            batch_size: Batch size for insertion

        Returns:
            Dictionary containing total insert count and list of inserted IDs
        """
        if not collection:
            collection = self.default_collection
        embeddings = [chunk.embedding for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        references_list = [chunk.metadata.get("title", "") for chunk in chunks]
        reference_ids_list = [chunk.metadata.get("article_id", 0) for chunk in chunks]
        keywords_list = [chunk.metadata.get("keywords", []) for chunk in chunks]
        authors_list = [chunk.metadata.get("authors", []) for chunk in chunks]
        author_ids_list = [chunk.metadata.get("author_ids", []) for chunk in chunks]
        base_ids_list = [chunk.metadata.get("base_ids", []) for chunk in chunks]
        corresponding_authors_list = [
            chunk.metadata.get("corresponding_authors", []) for chunk in chunks
        ]
        corresponding_author_ids_list = [
            chunk.metadata.get("corresponding_author_ids", []) for chunk in chunks
        ]
        impact_factor_list = [chunk.metadata.get("impact_factor", 0) for chunk in chunks]
        rbase_factor_list = [chunk.metadata.get("rbase_factor", 0) for chunk in chunks]
        pubdate_list = [int(chunk.metadata.get("pubdate", 0)) for chunk in chunks]

        datas = [
            {
                "embedding": embedding,
                "text": text,
                "reference": reference,
                "reference_id": reference_id,
                "keywords": keywords,
                "authors": authors,
                "author_ids": author_ids,
                "corresponding_authors": corresponding_authors,
                "corresponding_author_ids": corresponding_author_ids,
                "impact_factor": impact_factor,
                "pubdate": pubdate,
                "rbase_factor": rbase_factor,
                "base_ids": base_ids,
            }
            for embedding, text, reference, reference_id, keywords, authors, author_ids, corresponding_authors, corresponding_author_ids, impact_factor, pubdate, rbase_factor, base_ids in zip(
                embeddings,
                texts,
                references_list,
                reference_ids_list,
                keywords_list,
                authors_list,
                author_ids_list,
                corresponding_authors_list,
                corresponding_author_ids_list,
                impact_factor_list,
                pubdate_list,
                rbase_factor_list,
                base_ids_list,
            )
        ]
        batch_datas = [datas[i : i + batch_size] for i in range(0, len(datas), batch_size)]

        # Initialize result summary
        total_result = {"insert_count": 0, "ids": []}

        try:
            for batch_data in batch_datas:
                res = self.client.insert(collection_name=collection, data=batch_data)
                # Aggregate results
                if res:
                    total_result["insert_count"] += res.get("insert_count", 0)
                    if "ids" in res:
                        # Check and process IDs appropriately
                        total_result["ids"].extend(list(res["ids"]))
            # Return aggregated results
            return total_result
        except Exception as e:
            log.critical(f"fail to insert data, error info: {e}")
            return {"insert_count": 0, "ids": []}

    def search_data(
        self,
        collection: Optional[str],
        vector: Union[np.array, List[float]],
        top_k: int = 5,
        filter: Optional[str] = "",
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Search for most similar vectors in the database.

        Args:
            collection: Collection name
            vector: Query vector
            top_k: Number of most similar results to return
            filter: Query filter expression in Milvus syntax

        Returns:
            List of RetrievalResult objects containing search results
        """
        if not collection:
            collection = self.default_collection
        try:
            search_results = self.client.search(
                collection_name=collection,
                data=[vector],
                limit=top_k,
                filter=filter,
                output_fields=[
                    "embedding",
                    "text",
                    "reference",
                    "reference_id",
                    "pubdate",
                    "impact_factor",
                ],
                timeout=10,
            )

            return [
                RetrievalResult(
                    embedding=b["entity"]["embedding"],
                    text=b["entity"]["text"],
                    reference=b["entity"]["reference"],
                    score=b["distance"],
                    metadata={
                        "reference_id": b["entity"]["reference_id"],
                        "pubdate": b["entity"]["pubdate"],
                        "impact_factor": b["entity"]["impact_factor"],
                    },
                )
                for a in search_results
                for b in a
            ]
        except Exception as e:
            log.critical(f"fail to search data, error info: {e}")
            return []

    def list_collections(self, *args, **kwargs) -> List[CollectionInfo]:
        """
        List all collections in the database.

        Returns:
            List of CollectionInfo objects containing collection details
        """
        collection_infos = []
        try:
            collections = self.client.list_collections()
            for collection in collections:
                description = self.client.describe_collection(collection)
                collection_infos.append(
                    CollectionInfo(
                        collection_name=collection,
                        description=description["description"],
                    )
                )
        except Exception as e:
            log.critical(f"fail to list collections, error info: {e}")
        return collection_infos

    def clear_db(self, collection: str = "deepsearcher", *args, **kwargs):
        """
        Clear the specified collection from the database.

        Args:
            collection: Name of the collection to clear
        """
        if not collection:
            collection = self.default_collection
        try:
            self.client.drop_collection(collection)
        except Exception as e:
            log.warning(f"fail to clear db, error info: {e}")

    def delete_data(self, collection: str, ids: Optional[List[int]] = None, filter: Optional[str] = None, *args, **kwargs) -> int:
        """
        Delete data from the specified collection based on the filter.

        Args:
            collection: Collection name
            filter: Filter expression in Milvus syntax
            ids: List of IDs to delete

        Returns:
            Number of deleted documents
        """
        if not ids and not filter:
            log.warning("no ids or filter provided, skip delete")
            return 0

        try:
            if ids:
                rt = self.client.delete(collection_name=collection, filter=filter, ids=ids)
            else:
                rt = self.client.delete(collection_name=collection, filter=filter)
            return rt.get("delete_count", 0)
        except Exception as e:
            log.critical(f"fail to delete data, error info: {e}")
    
    def flush(self, collection_name: str, **kwargs):
        timeout = kwargs.get("timeout", None)
        self.client.flush(collection_name, timeout)
        
    def close(self):
        self.client.close()