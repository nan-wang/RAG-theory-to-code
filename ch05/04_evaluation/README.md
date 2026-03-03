# 效果评估

## 简介

本章构建了一套完整的 RAG 系统评估流水线。流程包括三个阶段：合成评估数据的生成（自动生成问答对并进行验证与改写）、RAG 系统的检索与生成指标计算（精确率、召回率、忠实度、幻觉率等），以及全局指标的汇总。评估流程通过 Shell 脚本进行编排，支持端到端的自动化评估。

## 环境配置检查

```bash
cd ch05/04_evaluation
python test_env_setup.py
```

## 运行步骤

### 生成评估数据

```bash
bash generate_synthetic_qa_pairs.sh --index-dir ../03_langchain_core_and_langgraph/09_naive_rag/data_chroma --collection-name olympic_games --output-dir data_eval/demo --num-generate 8
```

在这个过程中，`generate_synthetic_qa_pairs.sh`脚本会依次执行以下Python脚本，

- `python 03_generate_qa_pairs.py --index_dir ../03_langchain_core_and_langgraph/09_naive_rag/data_chroma  --collection_name olympic_games --output_dir data_eval/demo --num_docs 8`
- `python 03_validate_qa_pairs.py --input_path data_eval/demo/qa_pairs.raw.json --output_path data_eval/demo/qa_pairs.validated.json`
- `python 03_rewrite_qa_pairs.py --input_path data_eval/demo/qa_pairs.validated.json --output_dir data_eval/demo`

### 抽取标准答案的信息点
使用``qa_pairs.rewritten.json``，对标准答案抽取细颗粒度关键信息，保存在``data_eval/demo/keypoints.json``中。

```bash
python 01_extract_keypoints.py --ground-truth --output_path data_eval/demo/ data_eval/demo/qa_pairs.rewritten.json
```

### 测试评估数据
使用目标RAG系统对评估数据进行回答，结果保存在``data_eval/naive_rag/response.json``。为了保持标准答案的细颗粒度关键信息，使用上一步得到的输出文件``data_eval/demo/keypoints.json``作为输入。

```bash
python naive_rag.py --index --query --index_dir ../03_langchain_core_and_langgraph/09_naive_rag/data_chroma  --collection_name olympic_games --output_dir data_eval/naive_rag data_eval/demo/keypoints.json
```


### 评估效果

```bash
bash evaluate.sh --input-dir ./data_eval/naive_rag --num-docs 6
```

这个过程中，依次运行了以下代码，

- `python 01_extract_keypoints.py --response --output_path data_eval/naive_rag/ data_eval/naive_rag/response.json`。从生成的回答中冲去细颗粒度关键信息，结果保存在`data_eval/naive_rag/keypoints.json`，用于后续评估。
- `python 02_calculate_global_metrics.py --num_docs 6 --precision --recall --output_path data_eval/naive_rag data_eval/naive_rag/keypoints.json`。使用第一步输出的`data_eval/naive_rag/keypoints.json`作为输入，计算整体精确率、召回率、F1分数，结果保存在`data_eval/naive_rag/metrics/global_precision.json`和`data_eval/naive_rag/metrics/global_recall.json`
- `02_calculate_retrieval_metrics.py --num_docs 6 --precision --recall --output_path data_eval/naive_rag data_eval/naive_rag/keypoints.json`。使用第一步输出的`data_eval/naive_rag/keypoints.json`作为输入，计算检索模块的上下文精度和关键点召回率，结果保存在`data_eval/naive_rag/metrics/retrieval_context_precision.json`和`data_eval/naive_rag/metrics/retrieval_keypoints_recall.json`
- `python 02_calculate_generation_metrics.py --num_docs 6 --loyalty --hallucination --noise-sensitivity --context-utility-ratio --output_path data_eval/naive_rag data_eval/naive_rag/keypoints.json`。使用第一步输出的`data_eval/naive_rag/keypoints.json`作为输入，计算生成模块的忠诚度、幻觉度、噪声敏感度和上下文利用率，结果保存在`data_eval/naive_rag/metrics/generation_loyalty.json`、`data_eval/naive_rag/metrics/generation_hallucination.json`、`data_eval/naive_rag/metrics/generation_noise_sensitivity`、`data_eval/naive_rag/metrics/generation_context_utility_ratio.json`
 
    

## 内容提纲

| 标题 | 描述 | 文件 |
| --- | --- | --- |
| 5.4.1 细颗粒度关键信息 | 从问答对中提取关键点用于评估 | [01_extract_keypoints.py](01_extract_keypoints.py) |
| 5.4.2 计算全局指标 | 计算全局的精确率和召回率 | [02_calculate_global_metrics.py](02_calculate_global_metrics.py) |
| 5.4.2 计算检索指标 | 计算检索模块的精确率和召回率 | [02_calculate_retrieval_metrics.py](02_calculate_retrieval_metrics.py) |
| 5.4.2 计算生成指标 | 计算生成模块的忠实度、幻觉率、噪声敏感度等指标 | [02_calculate_generation_metrics.py](02_calculate_generation_metrics.py) |
| 5.4.3 生成合成问答对 | 基于文档内容自动生成问答对 | [03_generate_qa_pairs.py](03_generate_qa_pairs.py) |
| 5.4.3 校验问答对 | 对生成的问答对进行质量验证 | [03_validate_qa_pairs.py](03_validate_qa_pairs.py) |
| 5.4.3 重写写问答对 | 对验证通过的问答对进行改写优化 | [03_rewrite_qa_pairs.py](03_rewrite_qa_pairs.py) |
| 数据模型 | 定义 KeyPoint 等评估数据模型 | [datamodels.py](datamodels.py) |
| 工具函数 | 文档加载、分块等通用工具函数 | [utils.py](utils.py) |
| 合成数据提示词 | 生成合成数据的提示词模板 | [synthetic_data_prompt.py](prompts/synthetic_data_prompt.py) |
| 合成数据提示词（精简版） | 精简版的合成数据提示词 | [synthetic_data_prompt.lite.py](prompts/synthetic_data_prompt.lite.py) |
| 关键点提取提示词 | 用于关键点提取的提示词模板 | [keypoints_extract_prompt.py](prompts/keypoints_extract_prompt.py) |
| 回答关键点验证提示词 | 验证回答是否覆盖关键点的提示词 | [answer_keypoints_verify_prompt.py](prompts/answer_keypoints_verify_prompt.py) |
| 标准答案关键点验证提示词 | 验证标准答案关键点覆盖情况的提示词 | [groundtruth_keypoints_verify_prompt.py](prompts/groundtruth_keypoints_verify_prompt.py) |
| 改写问题提示词 | 用于改写问题的提示词模板 | [rewrite_question_prompt.py](prompts/rewrite_question_prompt.py) |
| 验证问答提示词 | 用于验证问答对质量的提示词 | [validate_question_answer_prompt.py](prompts/validate_question_answer_prompt.py) |
| 生成合成数据脚本 | 编排合成数据生成流程的 Shell 脚本 | [generate_synthetic_qa_pairs.sh](generate_synthetic_qa_pairs.sh) |
| 评估脚本 | 编排评估流程的 Shell 脚本 | [evaluate.sh](evaluate.sh) |
