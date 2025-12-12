from src.core import Agent
from src.tools import add, count_letter_in_string, compare, get_current_datetime, search_wikipedia, get_current_temperature

from openai import OpenAI

import os

API_KEY=os.getenv("SILICONFLOW_API_KEY")
BASE_URL=os.environ.get("SILICONFLOW_BASE_URL","https://api.siliconflow.cn/v1")

if __name__ == "__main__":
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'
    
    agent = Agent(
        client=client,
        model=MODEL_NAME,
        tools=[get_current_datetime, search_wikipedia, get_current_temperature],
    )

    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答


