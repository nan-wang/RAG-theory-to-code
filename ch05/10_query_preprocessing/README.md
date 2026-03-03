# 查询预处理

## 简介

本章实现三种查询预处理技术，用于在检索之前对用户查询进行优化。包括查询分解（将复杂问题拆解为多个可独立回答的子问题）、回退提示（将具体问题泛化为更易检索的通用问题），以及查询路由（根据问题类型将查询路由到最合适的数据源）。

## 环境配置检查

```bash
cd ch05/10_query_preprocessing
python test_env_setup.py
```

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| 查询分解 | 将复杂问题拆解为多个子问题，分别检索后合并生成回答 | [01_query_decomposition.py](01_query_decomposition.py) |
| 回退提示 | 使用 Few-shot 示例将具体问题改写为更通用的回退问题，提升检索召回率 | [03_query_stepback.py](03_query_stepback.py) |
| 查询路由 | 使用结构化输出将用户查询路由到最相关的数据源 | [04_query_routing.py](04_query_routing.py) |

## 运行步骤

在运行查询脚本之前，确保已通过索引脚本构建好向量数据库（`data_chroma/`）。

**查询分解**：将复杂问题自动拆解为若干子问题后分别检索。

```bash
python 01_query_decomposition.py \
  --index_dir ./data_chroma \
  --collection_name olympic_games \
  --question "过去5届夏季奥运会都是在哪里举办的?"
```

**回退提示**：将具体问题改写为更通用的背景问题后检索。

```bash
python 03_query_stepback.py \
  --index_dir ./data_chroma \
  --collection_name olympic_games \
  --question "北京申办过几次奥运会?"
```

**查询路由**：根据问题意图路由到最合适的数据源（无需向量数据库）。

```bash
python 04_query_routing.py \
  --question "里约奥运会哪个国家获得的金牌最多?"
```
