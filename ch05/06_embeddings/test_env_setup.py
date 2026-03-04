"""
环境变量检查脚本

运行方式：
  cd ch05/06_embeddings
  python test_env_setup.py
"""

import os
import sys

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


# 加载 .env
try:
    import dotenv

    dotenv.load_dotenv()
except ImportError:
    pass

print("\n检查环境变量（06_embeddings）\n")

REQUIRED = [
    ("OPENAI_API_KEY", "LLM API 密钥"),
    ("OPENAI_API_BASE", "LLM API 地址"),
]

OPTIONAL = [
    ("JINA_API_KEY", "Jina AI（Jina Embeddings）"),
    ("SILICONFLOW_API_KEY", "硅基流动（BGE-M3 嵌入模型）"),
    ("SILICONFLOW_BASE_URL", "硅基流动 API 地址"),
]

for var, desc in REQUIRED:
    val = os.getenv(var)
    if val:
        ok(f"{var}={val[:8]}***（{desc}）")
    else:
        fail(f"{var} 未设置（{desc}）")

for var, desc in OPTIONAL:
    val = os.getenv(var)
    if val:
        ok(f"{var}={val[:8]}***（{desc}）")
    else:
        warn(f"{var} 未设置（{desc}，按需配置）")

# 检查 LLM API 连通性
print()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
if api_key and base_url:
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", max_tokens=16)
        response = llm.invoke("你好")
        if response.content:
            ok(f"LLM API 调用成功")
        else:
            fail("LLM API 返回空内容")
    except Exception as e:
        fail(f"LLM API 调用失败：{e}")
else:
    warn("跳过 LLM API 连通性检查")

# 汇总
print(f"\n{'=' * 50}")
print(
    f"  {GREEN}通过：{passed}{RESET}  {RED}失败：{failed}{RESET}  {YELLOW}警告：{warned}{RESET}\n"
)
sys.exit(1 if failed else 0)
