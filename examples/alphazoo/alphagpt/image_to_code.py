import os
import base64
import requests
from dotenv import load_dotenv, find_dotenv
from pathlib import Path  # Python 3.6+ only

# 一、自动搜索 .env 文件
load_dotenv(verbose=True)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
def gpt_request(model, page):

  if model == 'gpt-4o':
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = 'gpt-4o'
  elif model == 'kimi':
    api_key=os.getenv('refresh_token')
    base_url='http://127.0.0.1:8000'
  else:
    raise NotImplementedError

  base64_image = encode_image('data/factorcalander/factorcalander2024_page{}.png'.format(page))

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": f"{model}",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "分析这张图片，给出图中因子的python计算代码，将代码封装为一个类，只输出代码片段，不要输出其他内容，确保输出内容可以直接存为py文件，直接开始回答"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
               # "url": "https://www.moonshot.cn/assets/logo/normal-dark.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 2048
  }

  response = requests.post("{}/v1/chat/completions".format(base_url), headers=headers, json=payload)

  code_snippet =  response.json()['choices'][0]['message']['content']
  with open('code_snippet/factorcalander_{}_{}.py'.format(model, page), 'w') as f:
    f.write(code_snippet)

if __name__ == '__main__':
  gpt_request(model='kimi', page=337)