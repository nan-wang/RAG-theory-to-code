"""MessagesPlaceholder - 在提示模板中插入消息列表

本脚本演示 LangChain 中 MessagesPlaceholder 的用法。
ChatPromptTemplate 用于构建聊天提示，其中 MessagesPlaceholder
可以在模板中预留一个"插槽"，在调用时动态插入多条消息。
这在需要传入对话历史（多轮对话）时非常有用。
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 构建聊天提示模板：
# - ("system", ...) 定义系统消息，设定 AI 的角色
# - MessagesPlaceholder("msgs") 预留一个名为 "msgs" 的消息插槽
prompt_template = ChatPromptTemplate(
    [("system", "你是一位乐于帮助的助理"), MessagesPlaceholder("msgs")]
)

# 调用模板时，通过 "msgs" 键传入消息列表
# HumanMessage 代表用户消息，AIMessage 代表 AI 回复
result = prompt_template.invoke(
    {
        "msgs": [
            HumanMessage(content="我是来自上海的开发者"),
            AIMessage(content="你好，今天有什么可以帮助你的?"),
        ]
    }
)

# result 是 ChatPromptValue 对象，包含完整的消息列表
print(type(result))
print(result)
