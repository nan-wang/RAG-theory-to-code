# 环境配置

## Python 版本

本项目需要 Python 3.10 或更高版本。

## 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 环境变量

大部分示例代码需要通过环境变量配置 API 密钥和相关参数。请在项目根目录下创建 `.env` 文件，并参考以下模板进行配置：

```bash
# LLM API（OpenAI 兼容接口）
OPENAI_API_KEY=<你的 API Key>
OPENAI_BASE_URL=<API 地址，例如 https://api.siliconflow.cn/v1>

# Jina AI（用于嵌入模型和重排序）
JINA_API_KEY=<你的 Jina API Key>

# 向量数据库（本地）
VECTOR_DB_DIR=./data_chroma
COLLECTION_NAME=olympic_games
EMBEDDING_MODEL=text-embedding-3-small

# 腾讯云向量数据库（生产环境）
DB_URL=<数据库地址>
DB_USERNAME=root
DB_KEY=<数据库密钥>

# Tavily（用于 Agentic RAG 中的网络搜索）
TAVILY_API_KEY=<你的 Tavily API Key>
```

> **提示**：不同章节的示例可能只需要部分环境变量，请根据具体示例的需求进行配置。
