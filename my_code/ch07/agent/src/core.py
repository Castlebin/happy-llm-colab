from openai import OpenAI
import json
from typing import List, Dict, Any
from utils import function_to_json
# 导入我们定义好的工具函数
from tools import *

import pprint

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

"""
Agent 的工作流程如下：
1. 接收用户输入
2. 调用大模型（如：Qwen），并告知它我们提供了哪些工具
3. 如果大模型决定调用工具，那么 Agent 会解析工具调用的信息，并执行相应的工具函数
4. Agent 将工具函数的结果返回给大模型，继续对话
5. 模型根据工具的结果生成最终回答
6. Agent 将最终回答返回给用户
"""
class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", 
                 tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        
        # 检查模型是否调用了工具
        if response.choices[0].message.tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content,
            })
            
            # 处理每个工具调用
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具
                tool_list = []
                tool_response = self.handle_tool_call(tool_call)
                # 将工具响应添加到消息列表中
                self.messages.append(tool_response)
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
                
            if self.verbose:
                print("工具调用:", response.choices[0].message.content, tool_list)
                
            # 再次获取模型的完成响应，这次包含工具调用的结果
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )
            
        # 将模型的最终响应添加到消息列表中
        self.messages.append({
            "role": "assistant",
            "content": final_response.choices[0].message.content,
        })
        
        return final_response.choices[0].message.content
    
    