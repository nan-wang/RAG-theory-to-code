# 部署

## 简介

本章演示如何将 RAG 系统部署到生产环境。采用前后端分离架构：后端使用 FastAPI 提供 RESTful API，前端使用 Streamlit 构建交互式聊天界面。RAG 流水线基于腾讯云向量数据库实现混合检索（向量 + BM25）和 Jina Reranker 重排序。支持 Docker 容器化一键部署。

## 使用步骤

### 创建腾讯云向量数据库实例
在使用腾讯云向量数据库之前，需要在腾讯云中创建一个向量数据库实例。向量数据库实例创建完成之后，在实例列表中，找到目标实例。选择目标实例 ID，单击“管理”按钮，进入实例详情页面。选择“密钥管理”选项，将密钥、用户名、外网URL保存在`api/src/.env`中。

```bash
$ cd ch05/12_deployment/api/src
$ cp .env.example .env
# 在.env中添加需要的LLM秘钥和Jina AI秘钥
$ python test_env_setup.py
```

```bash
# Jina AI
# 用于 JinaEmbeddings 向量检索
JINA_API_KEY=jina_xxx

# 腾讯云向量数据库
DB_URL=http://xxx.vectordb.tencent.com:8100
DB_USERNAME=root
DB_KEY=xxx
```

### 创建向量索引

```bash
$ cd ch05/12_deployment/api/src
$ python -m rag.index --index_input_dir ../../../data

# Loaded 24 documents
# [INFO] 已写入 32/338 条文档块
# ...
# [INFO] 已写入 338/338 条文档块
# [INFO] 索引完成：TencentVectorDB 中共存储 1014 条文档
```


### 本地测试后台服务运行

```bash
$ cd ch05/12_deployment/api/src
$ uvicorn main:app --host 0.0.0.0 --port 8000

# INFO:     Started server process [32723]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

在命令行中测试后台运行是否正常

```bash
$ curl -X 'POST' \
  'http://0.0.0.0:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "北京奥运会是哪一年"
}'

# {"selected_content":"[doc_1]article_title: 2008年夏季奥林匹克运动会 section_title: summary content: 第二十九届夏季奥林匹克运动会（英语：the Games of the XXIX Olympiad；法语：les Jeux de la 29e Olympiade），即2008年夏季奥运会，又称北京夏奥，是于2008年8月8日至24日在中华人民共和国首都北京市举行的综合运动会。","answer":"北京奥运会是在2008年举办的，具体时间为2008年8月8日至8月24日。"}
```

### 本地测试前端服务运行
确保后台服务已经在运行后，在命令行中测试前端服务。

> streamlit的命令行安装参考[官方文档](https://streamlit.io/#install)

```bash
$ cd ch05/12_deployment/frontend/src
$ streamlit run main.py

#  You can now view your Streamlit app in your browser.

#  Local URL: http://localhost:8501
#  Network URL: http://192.168.31.83:8501
```

打开浏览器，访问[http://localhost:8501](http://localhost:8501)。


### 本地测试Docker服务运行

> Docker安装请参考[官方文档](https://docs.docker.com/desktop/setup/install/mac-install/)

创建`ch05/12_deployment/.env`。

```bash
$ cd ch05/12_deployment
$ cp .env.example .env
# 在.env中添加需要的LLM秘钥和Jina AI秘钥
$ python test_env_setup.py
$ docker compose-up
```


## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| Docker Compose 配置 | 定义 API 和前端服务的容器编排配置 | [docker-compose.yml](docker-compose.yml) |
| API Dockerfile | 后端 API 服务的容器构建文件 | [api/Dockerfile](api/Dockerfile) |
| API 依赖配置 | 后端 API 的 Python 依赖配置 | [api/pyproject.toml](api/pyproject.toml) |
| API 入口 | FastAPI 应用入口，定义 `/ask` 接口 | [api/src/main.py](api/src/main.py) |
| API 启动脚本 | API 服务的容器启动脚本 | [api/src/entrypoint.sh](api/src/entrypoint.sh) |
| 请求模型 | 定义 API 请求的数据模型 | [api/src/models/query.py](api/src/models/query.py) |
| RAG 索引构建 | 构建腾讯云向量数据库的混合检索索引 | [api/src/rag/rag_index.py](api/src/rag/rag_index.py) |
| RAG 查询流程 | 基于 LangGraph 实现的 RAG 查询流水线 | [api/src/rag/rag_query.py](api/src/rag/rag_query.py) |
| 混合检索实现 | 腾讯云向量数据库的混合检索封装 | [api/src/rag/tcvectordb_hybrid_search.py](api/src/rag/tcvectordb_hybrid_search.py) |
| 生成提示词 | RAG 生成环节的提示词模板 | [api/src/rag/prompts.py](api/src/rag/prompts.py) |
| RAG 工具函数 | 文档加载、分块等工具函数 | [api/src/rag/utils.py](api/src/rag/utils.py) |
| 异步工具 | 异步执行的工具函数 | [api/src/utils/async_utils.py](api/src/utils/async_utils.py) |
| 前端 Dockerfile | 前端服务的容器构建文件 | [frontend/Dockerfile](frontend/Dockerfile) |
| 前端依赖配置 | 前端的 Python 依赖配置 | [frontend/pyproject.toml](frontend/pyproject.toml) |
| 前端主页面 | Streamlit 聊天界面的实现 | [frontend/src/main.py](frontend/src/main.py) |
| 前端启动脚本 | 前端服务的容器启动脚本 | [frontend/src/entrypoint.sh](frontend/src/entrypoint.sh) |
