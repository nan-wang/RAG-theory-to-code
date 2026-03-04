"""
环境变量检查脚本

运行方式：
  cd ch05/12_deployment
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

print("\n检查环境变量（12_deployment）\n")

REQUIRED = [
    ("OPENAI_API_KEY", "LLM API 密钥"),
    ("OPENAI_API_BASE", "LLM API 地址"),
    ("JINA_API_KEY", "Jina AI（向量检索和重排序）"),
    ("DB_URL", "腾讯云向量数据库地址"),
    ("DB_KEY", "腾讯云向量数据库密钥"),
    ("DB_USERNAME", "腾讯云向量数据库用户名"),
    ("CHATBOT_URL", "前端调用后端 API 的地址"),
]

OPTIONAL = [
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

# 汇总
print(f"\n{'=' * 50}")
print(f"  {GREEN}通过：{passed}{RESET}  {RED}失败：{failed}{RESET}  {YELLOW}警告：{warned}{RESET}\n")
sys.exit(1 if failed else 0)
