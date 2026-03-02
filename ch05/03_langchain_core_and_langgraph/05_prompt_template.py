"""PromptTemplate - 字符串提示模板与变量替换

本脚本演示 LangChain 中 PromptTemplate 的基本用法。
PromptTemplate 使用花括号 {variable} 定义模板变量，
调用 invoke 时传入变量值即可生成完整的提示文本。
适用于构建简单的、非对话式的提示。
"""

from langchain_core.prompts import PromptTemplate

# from_template 方法自动解析模板中的 {location} 变量
prompt_template = PromptTemplate.from_template("{location}的天气怎么样？")

# invoke 传入变量字典，返回 StringPromptValue 对象
prompt = prompt_template.invoke({"location": "北京"})

print(prompt)
# 输出: text='北京的天气怎么样？'
print(type(prompt))
# 输出: <class 'langchain_core.prompt_values.StringPromptValue'>
