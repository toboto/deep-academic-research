from pymilvus import MilvusClient, DataType
from deepsearcher.embedding.milvus_embedding import MilvusEmbedding
import random

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection = "rbase_articles"
description = "Academic Article Dataset"
has_collection = client.has_collection(collection, timeout=5)
if has_collection:
    print(f"Collection {collection} already exists")
else:
    schema = client.create_schema(
    enable_dynamic_field=False, auto_id=True, description=description
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="L2")
    client.create_collection(
    collection,
    schema=schema,
    index_params=index_params,
    consistency_level="Strong",
    )

data=[
    {"embedding": [0.1 for _ in range(768)], "text": "Hello, world! No. 1"},
    {"embedding": [0.2 for _ in range(768)], "text": "Hello, world! No. 2"},
    {"embedding": [0.3 for _ in range(768)], "text": "Hello, world! No. 3"},
]

# res = client.insert(
#     collection_name="milvus_entities",
#     data=data
# )
# 
# print(res)


embedding = MilvusEmbedding()
search_vector = embedding.embed_documents(["two MRM models"])

res = client.search(
    collection_name=collection,
    data=search_vector,
    # filter="ARRAY_CONTAINS(keywords, 'Europe')",
    # filter="reference_id == 28839",
    output_fields=["text", "reference", "reference_id", "metadata"],
    limit=10
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)