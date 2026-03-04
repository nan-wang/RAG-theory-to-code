import argparse
import os
import time
from pathlib import Path

import dotenv
import requests
import tcvectordb
from tcvdb_text.encoder import BM25Encoder
from tcvectordb.model.enum import ReadConsistency, FieldType, IndexType, MetricType
from tcvectordb.model.index import (
    Index,
    FilterIndex,
    VectorIndex,
    SparseIndex,
    HNSWParams,
)

from rag.utils import load_documents, split_sections, split_chunks

dotenv.load_dotenv()


def get_all_splits(input_dir):
    """加载并切分文档，返回所有文本块。

    Args:
        input_dir: 文档所在文件夹路径，自动匹配其中所有 .txt 文件

    Returns:
        chunks: 切分后的文本块列表
    """
    glob_pattern = str(Path(input_dir) / "*.txt")
    docs = load_documents(glob_pattern)
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


def encode_texts(texts, jina_api_key="", batch_size=32):
    """调用 Jina API 批量生成稠密向量。

    Args:
        texts: 待编码的文本列表
        jina_api_key: Jina API 密钥（优先从环境变量读取）
        batch_size: 每批处理的文本数量

    Returns:
        embeddings: 与 texts 等长的向量列表
    """
    url = "https://api.jina.ai/v1/embeddings"
    jina_api_key = os.environ.get("JINA_API_KEY", jina_api_key)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {jina_api_key}",
            "Accept-Encoding": "identity",
            "Content-type": "application/json",
        }
    )
    embeddings = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        data = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": batch,
        }
        resp = session.post(url, json=data).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])
        cur_embeddings = resp["data"]
        sorted_embeddings = sorted(cur_embeddings, key=lambda e: e["index"])  # type: ignore
        batch_embeddings = [result["embedding"] for result in sorted_embeddings]
        embeddings.extend(batch_embeddings)
    return embeddings


def main():
    """主流程：解析参数 → 建库 → 编码 → 写入 → 验证。"""
    parser = argparse.ArgumentParser(description="构建混合索引并写入 TencentVectorDB")
    parser.add_argument(
        "--index_input_dir",
        default="../../../../data",
        help="文档所在文件夹路径，自动索引其中所有 .txt 文件，默认为 ../../../../data",
    )
    args = parser.parse_args()

    DB_URL = os.environ.get("DB_URL", "")
    DB_USERNAME = os.environ.get("DB_USERNAME", "root")
    DB_KEY = os.environ.get("DB_KEY", "")
    DB_NAME = "db-olympic-games"
    COLLECTION_NAME = "olympic-games-hybrid"

    # 连接 TencentVectorDB
    client = tcvectordb.RPCVectorDBClient(
        url=DB_URL,
        key=DB_KEY,
        username=DB_USERNAME,
        read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
        timeout=30,
    )

    # 获取或创建数据库
    db_exists = False
    for db in client.list_databases():
        if db.database_name == DB_NAME:
            db_exists = True
            break
    if not db_exists:
        db = client.create_database(database_name=DB_NAME)
    else:
        db = client.database(DB_NAME)

    # 定义索引结构（稠密向量 + 稀疏向量 + 过滤字段）
    index = Index(
        FilterIndex("id", FieldType.String, IndexType.PRIMARY_KEY),
        VectorIndex(
            "vector",
            dimension=1024,
            index_type=IndexType.HNSW,
            metric_type=MetricType.COSINE,
            params=HNSWParams(m=16, efconstruction=200),
        ),
        SparseIndex(
            name="sparse_vector",
            field_type=FieldType.SparseVector,
            index_type=IndexType.SPARSE_INVERTED,
            metric_type=MetricType.IP,
        ),
        FilterIndex("text", FieldType.String, IndexType.FILTER),
        FilterIndex("source", FieldType.String, IndexType.FILTER),
        FilterIndex("title", FieldType.String, IndexType.FILTER),
    )

    # 获取或创建集合
    try:
        db.describe_collection(COLLECTION_NAME)
    except tcvectordb.exceptions.VectorDBException:
        db.create_collection(
            name=COLLECTION_NAME,
            shard=1,
            replicas=0,
            description="this is a collection olympic games",
            index=index,
        )

    # 加载并切分文档
    chunks = get_all_splits(args.index_input_dir)
    bm25 = BM25Encoder.default("zh")
    batch_size = 32
    total_count = len(chunks)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.id for chunk in chunks]

    # BM25 稀疏编码 + Jina 稠密编码
    sparse_embeddings = bm25.encode_texts(texts)
    dense_embeddings = encode_texts(texts, batch_size=batch_size)

    # 批量写入 TencentVectorDB
    for start in range(0, total_count, batch_size):
        end = min(start + batch_size, total_count)
        docs = []
        for id in range(start, end, 1):
            if metadatas[id] and isinstance(metadatas[id], dict):
                metadata = metadatas[id]
            else:
                metadata = {}
            doc_id = ids[id] or "{}-{}-{}".format(time.time_ns(), hash(texts[id]), id)
            doc_attrs = {
                "id": doc_id,
                "vector": dense_embeddings[id],
                "sparse_vector": sparse_embeddings[id],
                "text": texts[id],
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
            }
            docs.append(doc_attrs)
        client.upsert(
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            documents=docs,
            timeout=30,
        )
        print(f"[INFO] 已写入 {end}/{total_count} 条文档块")

    # 查询实际存储数量并输出
    try:
        count_result = client.count(
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
        )
        print(f"[INFO] 索引完成：TencentVectorDB 中共存储 {count_result} 条文档")
    except Exception:
        print(f"[INFO] 索引完成：本次写入 {total_count} 条文档块")

    client.close()


if __name__ == "__main__":
    main()
