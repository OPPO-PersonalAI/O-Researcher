#!/usr/bin/env python3
"""
快速测试 CrawlPage 服务器

使用方法：
python test_crawl_page_simple.py <endpoint_url>
e.g. python test_crawl_page_simple.py http://127.0.0.1:20001/crawl_page
"""

import argparse
import json
import os
import time

import requests

from dotenv import load_dotenv

# 创建参数解析器
parser = argparse.ArgumentParser(description='Crawl page test script.')
parser.add_argument('endpoint_url', type=str, help='The endpoint URL to test against (e.g., http://127.0.0.1:20001/crawl_page)')

# 解析命令行参数
args = parser.parse_args()


test_data = {
    "urls": ["https://en.wikipedia.org/wiki/Qwen", "https://en.wikipedia.org/wiki/Alibaba_Cloud"],
    "task": "what is qwen?", 
    "web_search_query": "qwen is developed by?",
    "think_content": "to think...",
}

all_data = [test_data]

for data in all_data:
    print("\n" + "="*20)
    print(f"Testing Summary task: {data.get('task')}")
    print(f"Testing Summary web_search_query: {data.get('web_search_query')}")
    print(f"Testing Summary think_content: {data.get('think_content')}")
    print("="*20)
    try:
        # 发送请求
        url = args.endpoint_url
        # url = "http://10.77.226.105:30001/crawl_page"
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        # 检查HTTP状态码
        response.raise_for_status()
        
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"Response content: {response.text}")  # Print first 500 chars of response
            continue

        # 输出结果
        if result.get("success"):
            print("成功!")
            print(f"处理时间: {result.get('processing_time'):.1f}秒")
            print("\n结果:")
            print("-" * 50)
            print(result.get('obs'))
            print("-" * 50)
        else:
            print(f"失败: {result.get('error_message', '未知错误')}")
            
    except requests.exceptions.ConnectionError:
        print("连接错误: 无法连接到服务器，请确保服务器正在运行")
        break # Stop testing if connection fails
    except requests.exceptions.Timeout:
        print("超时错误: 请求超时")
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {str(e)}")
    except Exception as e:
        print(f"未知错误: {str(e)}")

print("\n所有测试完成")