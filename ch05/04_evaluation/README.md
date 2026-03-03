# 效果评估

## 简介

本章构建了一套完整的 RAG 系统评估流水线。流程包括三个阶段：合成评估数据的生成、细颗粒度关键信息抽取和指标计算。

## 目录结构

```
04_evaluation/
├── .env / .env.example          # 共享环境变量
├── README.md
│
├── 01_extract_keypoints/        # 细颗粒度关键信息抽取
│   ├── naive_rag.py             # 索引构建与检索问答
│   ├── 01_extract_keypoints.py  # 关键点抽取
│   ├── utils.py                 # 文档加载、分块工具函数
│   ├── datamodels.py            # KeyPoints 数据模型
│   └── prompts/
│       └── keypoints_extract_prompt.py
│
├── 02_calculate_metrics/        # 指标计算
│   ├── 02_calculate_global_metrics.py      # 全局精确率、召回率
│   ├── 02_calculate_retrieval_metrics.py   # 检索模块指标
│   ├── 02_calculate_generation_metrics.py  # 生成模块指标
│   ├── utils.py                 # 指标持久化、关键点验证工具函数
│   ├── datamodels.py            # KeyPoint 数据模型
│   └── prompts/
│       ├── groundtruth_keypoints_verify_prompt.py
│       └── answer_keypoints_verify_prompt.py
│
└── 03_generate_qa_pairs/        # 合成问答对生成
    ├── 03_generate_qa_pairs.py  # 生成问答对
    ├── 03_validate_qa_pairs.py  # 校验问答对
    ├── 03_rewrite_qa_pairs.py   # 重写问答对
    └── prompts/
        ├── synthetic_data_prompt.py
        ├── validate_question_answer_prompt.py
        └── rewrite_question_prompt.py
```

## 环境配置检查

每个子目录都包含独立的 `.env`、`.env.example` 和 `test_env_setup.py`：

```bash
cd ch05/04_evaluation/01_extract_keypoints
python test_env_setup.py
```

## 运行步骤

### 生成评估数据

#### [可选]构建索引
构建索引，用于抽取文本块。可以根据自己的实际需求修改文本块切分的方式。

```bash
cd ch05/04_evaluation/01_extract_keypoints
python naive_rag.py --index --no-query --index_dir ../data_chroma  --collection_name olympic_games --index_input_dir ../../data
```

#### 生成评估数据

从索引中随机抽取文本块，用于生成评估数据。

```bash
cd ch05/04_evaluation/
python 03_generate_qa_pairs/03_generate_qa_pairs.py --index_dir data_chroma --collection_name olympic_games --output_dir data_eval --num_docs 2
```

读取QA对，使用大模型进行验证，将验证结果保存在`data_eval/qa_pairs.validated.json`

```bash
python 03_generate_qa_pairs/03_validate_qa_pairs.py --input_fn data_eval/qa_pairs.raw.json --output_dir data_eval
```

读取QA对，过滤掉不符合要求的部分。对于剩余的问答对，使用大模型进行重写，将重写的结果保存在`data_eval/qa_pairs.rewritten.json`

```bash
python 03_generate_qa_pairs/03_rewrite_qa_pairs.py --input_fn data_eval/qa_pairs.validated.json --output_dir data_eval
```


### 抽取细颗粒度关键信息

#### [可选]测试评估数据
使用目标RAG系统对评估数据进行回答，结果保存在``naive_rag/response.json``。

```bash
cd ch05/04_evaluation/01_extract_keypoints
python naive_rag.py --query --index_dir ../data_chroma  --collection_name olympic_games --output_dir ../naive_rag --query_input_path ../data_eval/qa_pairs.rewritten.json
```

#### 抽取细颗粒度关键信息

从生成的回答和标准答案中抽取细颗粒度关键信息，结果保存在`naive_rag/keypoints.json`，用于后续评估。

```bash
python 01_extract_keypoints/01_extract_keypoints.py --ground-truth --response --output_dir naive_rag/ --input_fn naive_rag/response.json
```

### 评估效果

使用上一步输出的`naive_rag/keypoints.json`作为输入，计算整体精确率、召回率、F1分数

```bash
cd ch05/04_evaluation
python 02_calculate_metrics/02_calculate_global_metrics.py --num_docs 6 --precision --recall --output_dir naive_rag --input_fn naive_rag/keypoints.json
```

使用`naive_rag/keypoints.json`作为输入，计算检索模块的上下文精度和关键点召回率

```bash
python 02_calculate_metrics/02_calculate_retrieval_metrics.py --num_docs 2 --precision --recall --output_dir naive_rag --input_fn naive_rag/keypoints.json
```

使用`naive_rag/keypoints.json`作为输入，计算生成模块的忠诚度、幻觉度、噪声敏感度和上下文利用率

```bash
python 02_calculate_metrics/02_calculate_generation_metrics.py --num_docs 2 --loyalty --hallucination --noise-sensitivity --context-utility-ratio --output_dir naive_rag --input_fn naive_rag/keypoints.json
```

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| 5.4.1 细颗粒度关键信息 | 从问答对中提取关键点用于评估 | [01_extract_keypoints.py](01_extract_keypoints/01_extract_keypoints.py) |
| 5.4.2 计算全局指标 | 计算全局的精确率和召回率 | [02_calculate_global_metrics.py](02_calculate_metrics/02_calculate_global_metrics.py) |
| 5.4.2 计算检索指标 | 计算检索模块的精确率和召回率 | [02_calculate_retrieval_metrics.py](02_calculate_metrics/02_calculate_retrieval_metrics.py) |
| 5.4.2 计算生成指标 | 计算生成模块的忠实度、幻觉率、噪声敏感度等指标 | [02_calculate_generation_metrics.py](02_calculate_metrics/02_calculate_generation_metrics.py) |
| 5.4.3 生成合成问答对 | 基于文档内容自动生成问答对 | [03_generate_qa_pairs.py](03_generate_qa_pairs/03_generate_qa_pairs.py) |
| 5.4.3 校验问答对 | 对生成的问答对进行质量验证 | [03_validate_qa_pairs.py](03_generate_qa_pairs/03_validate_qa_pairs.py) |
| 5.4.3 重写问答对 | 对验证通过的问答对进行改写优化 | [03_rewrite_qa_pairs.py](03_generate_qa_pairs/03_rewrite_qa_pairs.py) |
| 关键点提取提示词 | 用于关键点提取的提示词模板 | [keypoints_extract_prompt.py](01_extract_keypoints/prompts/keypoints_extract_prompt.py) |
| 回答关键点验证提示词 | 验证回答是否覆盖关键点的提示词 | [answer_keypoints_verify_prompt.py](02_calculate_metrics/prompts/answer_keypoints_verify_prompt.py) |
| 标准答案关键点验证提示词 | 验证标准答案关键点覆盖情况的提示词 | [groundtruth_keypoints_verify_prompt.py](02_calculate_metrics/prompts/groundtruth_keypoints_verify_prompt.py) |
| 合成数据提示词 | 生成合成数据的提示词模板 | [synthetic_data_prompt.py](03_generate_qa_pairs/prompts/synthetic_data_prompt.py) |
| 改写问题提示词 | 用于改写问题的提示词模板 | [rewrite_question_prompt.py](03_generate_qa_pairs/prompts/rewrite_question_prompt.py) |
| 验证问答提示词 | 用于验证问答对质量的提示词 | [validate_question_answer_prompt.py](03_generate_qa_pairs/prompts/validate_question_answer_prompt.py) |
| 生成合成数据脚本 | 编排合成数据生成流程的 Shell 脚本 | [generate_synthetic_qa_pairs.sh](generate_synthetic_qa_pairs.sh) |
| 评估脚本 | 编排评估流程的 Shell 脚本 | [evaluate.sh](evaluate.sh) |
| Naive RAG | 统一的索引构建与检索问答脚本 | [naive_rag.py](01_extract_keypoints/naive_rag.py) |
