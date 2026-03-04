"""文档加载与分块工具函数，供提示词优化 RAG 流程使用。"""

import glob
import re
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_year(fn):
    """从文件名前四个字符解析并返回奥运年份（整数）。"""
    return int(Path(fn).stem[:4])


def get_season(fn):
    """从文件名第 6-7 个字符解析并返回赛季标识（如 'su' 或 'wi'）。"""
    return Path(fn).stem[5:7]


def load_documents(pathname: str, with_metadata=False):
    """按 glob 模式加载所有 txt 文件，可选附加年份和赛季元数据，返回文档列表。"""
    docs = []
    for file in glob.glob(pathname):
        loader = TextLoader(file)
        _docs = loader.load()
        if with_metadata:
            year = get_year(file)
            season = get_season(file)
            for doc in _docs:
                doc.metadata["year"] = year
                doc.metadata["season"] = season
                docs.append(doc)
        else:
            docs += _docs
    return docs


def format_docs(docs):
    """将文档列表格式化为带编号标签的字符串，供提示词使用。"""
    output_list = []
    for idx, doc in enumerate(docs):
        doc_str = doc.page_content.replace("\n", " ")
        output_list.append(f"[doc_{idx + 1}]{doc_str}")
    return "\n\n".join(output_list)


def split_sections(text, source=None, skip_empty_sections=False):
    """将 Wikipedia 格式文本按 == 标题标记切分为带层级元数据的章节文档列表。"""
    sections = []
    pattern = r"(==+)(.*?)==+\s*([^=]*)"

    # This dictionary helps to track the current section level and index
    section_counters = {1: -1, 2: -1, 3: -1}
    parent_title = ""
    prev_level = 0
    section_title = [
        "",
    ]
    text = f"== summary ==\n\n{text}"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        level = (
            len(match.group(1)) - 1
        )  # Determine the section level by the number of '='
        title = match.group(2).strip()
        content = match.group(3).strip()

        if prev_level == 0:
            section_title.append(title)
            prev_level = level
        else:
            if prev_level == level:
                # pop the last section title
                section_title.pop()
                # push the current section title
                section_title.append(title)
            elif prev_level < level:
                # set the parent section title
                section_title.append(title)
                prev_level = level
            elif prev_level > level:
                section_title.pop()
                for _ in range(prev_level - level):
                    section_title.pop()
                section_title.append(title)
                prev_level = level
        # Reset section index for the lower level when we encounter a higher-level section
        if level == 1:
            section_counters[2] = -1
        if level == 2:
            section_counters[3] = -1
        section_counters[level] += 1

        metadata = {
            "source": source,
            "title": title,
            "parent_section": "_".join(section_title[1:-1]),
            "section_level": level,
            "section_index": section_counters[level],
        }
        if title in [
            "注释",
            "参见",
            "参考文献",
            "外部链接",
            "奖牌榜",
            "比赛日程",
            "参考",
        ]:
            continue
        if skip_empty_sections and not content:
            continue
        sections.append(Document(page_content=content, metadata=metadata))
    return sections


def split_chunks(docs: Iterable[Document]):
    """将章节文档进一步切分为固定大小的文本块，并在内容前缀添加文章及章节标题。"""
    results = []
    # split the sections into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
        separators=["。", "！", "？", "\?", "\n\n", "\n", "\n\n\n"],
        is_separator_regex=True,
        keep_separator="end",
    )

    for chunk in text_splitter.split_documents(docs):
        content = chunk.page_content
        metadata = chunk.metadata
        section_title = metadata["title"]
        if metadata["parent_section"]:
            section_title = f"{metadata['parent_section']}_{section_title}"
        content = f"section_title: {section_title}\ncontent: {content}"
        article_title = Path(metadata.get("source", "")).name.removesuffix(".txt")
        content = f"article_title: {article_title}\n{content}"
        results.append(Document(page_content=content, metadata=metadata))

    return results
