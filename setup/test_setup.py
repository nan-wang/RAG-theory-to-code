"""
环境配置检查脚本

运行此脚本以验证你的开发环境是否正确配置：
  python setup/test_setup.py

检查内容：
  1. Python 版本（需要 3.10+）
  2. 核心依赖包是否已安装
  3. 示例数据是否存在
"""

import sys

# ── 颜色输出 ────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

passed = 0
failed = 0
warned = 0


def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}[OK]{RESET} {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  {RED}[FAIL]{RESET} {msg}")


def warn(msg):
    global warned
    warned += 1
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


# ── 1. Python 版本 ──────────────────────────────────────────────────

print("\n1. 检查 Python 版本")
v = sys.version_info
if (v.major, v.minor) >= (3, 10):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(f"Python {v.major}.{v.minor}.{v.micro}（需要 3.10+）")


# ── 2. 核心依赖包 ──────────────────────────────────────────────────

print("\n2. 检查核心依赖包")

CORE_PACKAGES = [
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_community", "langchain-community"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_chroma", "langchain-chroma"),
    ("langchain_text_splitters", "langchain-text-splitters"),
    ("langgraph", "langgraph"),
    ("openai", "openai"),
    ("chromadb", "chromadb"),
    ("dotenv", "python-dotenv"),
    ("pydantic", "pydantic"),
    ("tqdm", "tqdm"),
    ("numpy", "numpy"),
]

OPTIONAL_PACKAGES = [
    ("pkuseg", "pkuseg", "中文分词（BM25 检索）"),
    ("langchain_huggingface", "langchain-huggingface", "本地嵌入模型"),
    ("sentence_transformers", "sentence-transformers", "嵌入模型微调"),
    ("datasets", "datasets", "嵌入模型微调"),
    ("fastapi", "fastapi", "部署"),
    ("streamlit", "streamlit", "部署"),
]

for import_name, pip_name in CORE_PACKAGES:
    try:
        __import__(import_name)
        ok(pip_name)
    except ImportError:
        fail(f"{pip_name}（运行 pip install {pip_name}）")

print("\n   可选依赖包：")
for import_name, pip_name, desc in OPTIONAL_PACKAGES:
    try:
        __import__(import_name)
        ok(f"{pip_name}（{desc}）")
    except ImportError:
        warn(f"{pip_name} 未安装（{desc}，按需安装：pip install {pip_name}）")


# ── 3. 示例数据 ────────────────────────────────────────────────────

print("\n3. 检查示例数据")

from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / "ch05" / "data"
if data_dir.exists():
    txt_files = list(data_dir.glob("*.txt"))
    if txt_files:
        ok(f"找到 {len(txt_files)} 个数据文件（{data_dir}）")
    else:
        fail(f"数据目录为空（{data_dir}）")
else:
    fail(f"数据目录不存在（{data_dir}）")


# ── 汇总 ──────────────────────────────────────────────────────────

print(f"\n{'=' * 50}")
print(f"  {GREEN}通过：{passed}{RESET}  "
      f"{RED}失败：{failed}{RESET}  "
      f"{YELLOW}警告：{warned}{RESET}")

if failed == 0:
    print(f"\n  {GREEN}环境配置检查通过！{RESET}")
else:
    print(f"\n  {RED}请修复上述失败项后重新运行此脚本。{RESET}")

print(f"\n  提示：各章节的环境变量配置请在对应文件夹中运行：")
print(f"  python test_env_setup.py\n")

sys.exit(1 if failed else 0)
