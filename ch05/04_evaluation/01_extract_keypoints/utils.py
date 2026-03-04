"""文档加载、章节拆分和文本分块的工具函数。"""

import re
import copy
from pathlib import Path
import glob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_sections(content, skip_empty_sections=True):
    """将 Wiki 格式文本按 == 标题层级拆分为章节字典列表。"""
    sections = []
    pattern = r"(==+)(.*?)==+\s*([^=]*)"

    # This dictionary helps to track the current section level and index
    section_counters = {1: -1, 2: -1, 3: -1}
    parent_section = ""

    # 在文档开头插入虚拟的 summary 节，使正文内容也能被匹配到
    text = f"== summary ==\n\n{content}"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        level = (
            len(match.group(1)) - 1
        )  # Determine the section level by the number of '='
        title = match.group(2).strip()
        content = match.group(3).strip()

        if skip_empty_sections and not content:
            continue
        # Reset section index for the lower level when we encounter a higher-level section
        if level == 1:
            section_counters[2] = -1
            parent_section = ""

        section_counters[level] += 1

        if level == 2:
            parent_section = sections[-1][
                "title"
            ]  # The parent section is the last level 1 section

        section_info = {
            "title": title,
            "content": content,
            "parent_section": parent_section,
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
        sections.append(section_info)

    return sections


def create_section_documents(
    section_list, metadatas, add_section_title=True, add_article_title=True
):
    """将章节列表转换为带元数据的 LangChain Document 对象列表。"""
    _metadatas = metadatas
    documents = []
    for i, sec in enumerate(section_list):
        metadata = copy.deepcopy(_metadatas)
        metadata["section_index"] = i
        metadata["section_level"] = sec["section_level"]
        metadata["parent_section"] = sec["parent_section"]
        metadata["section_title"] = sec["title"]
        content = sec["content"]
        section_title = metadata["section_title"]
        if metadata["parent_section"]:
            section_title = f"{metadata['parent_section']}_{section_title}"
        if add_section_title:
            content = f"section_title: {section_title}\ncontent: {content}"
        if add_article_title:
            title = Path(metadata.get("source", "")).name.removesuffix(".txt")
            content = f"article_title: {title}\n{content}"
        new_doc = Document(page_content=content, metadata=metadata)
        documents.append(new_doc)
    return documents


def get_chunks(docs: list[Document]):
    """将文档列表先按章节拆分，再按固定大小切分为文本块。"""
    # split each document into sections
    section_docs = []
    for doc in docs:
        section_list = split_sections(doc.page_content)
        section_docs += create_section_documents(section_list, doc.metadata)

    # split the sections into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )

    return text_splitter.split_documents(section_docs)


def load_documents(pathname: str):
    """按 glob 模式加载所有匹配文件，返回 Document 列表。"""
    docs = []
    for file in glob.glob(pathname):
        loader = TextLoader(file)
        _docs = loader.load()
        docs += _docs
    return docs


def format_docs(docs):
    """将多个文档的内容以双换行符拼接为单个字符串。"""
    return "\n\n".join(doc.page_content for doc in docs)
