import os

API_KEY=os.getenv("DASHSCOPE_API_KEY")
BASE_URL=os.environ.get("DASHSCOPE_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1")


from openai import OpenAI

if __name__ == "__main__":
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    MODEL_NAME = 'qwen2.5-vl-32b-instruct'
    
    


