"""ChatModel 基础 - 使用 ChatOpenAI 调用大语言模型

本脚本演示如何通过 LangChain 的 ChatOpenAI 类调用不同的大语言模型。
ChatOpenAI 兼容所有 OpenAI API 格式的接口，通过 base_url 参数
可以切换到不同的模型提供商（如硅基流动 SiliconFlow）。
"""

from langchain_openai.chat_models import ChatOpenAI

# 示例1：调用 OpenAI 官方的 gpt-4o-mini 模型（需替换为真实 API Key）
llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-<YOUR_OPENAI_API_KEY>")
response = llm.invoke("你好！")
print(response.content)
# 输出：'你好！今天过得怎么样？有什么我可以帮助你的吗？'

# 示例2：调用硅基流动提供的 DeepSeek-R1 模型
# 通过 base_url 参数指定第三方 API 地址，实现模型提供商的切换
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1",
    api_key="sk-<YOUR_SILICONFLOW_API_KEY>",
    base_url="https://api.siliconflow.cn/v1",
)
response = llm.invoke("你好！")
print(response.content)
# 输出：'你好！很高兴见到你，有什么我可以帮忙的吗？'
