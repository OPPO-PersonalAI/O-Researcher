#!/usr/bin/env python3
"""
测试 Serper v3 服务, Serper v3 包括了 serper (with cache) + crawl_page + summary 功能集成
使用方法：
python test_cache_serper_server_v3.py <endpoint_url>
e.g. python test_cache_serper_server_v3.py http://127.0.0.1:9002/search
e.g. python test_cache_serper_server_v3.py http://10.236.17.172:9002/search
"""

import argparse
import json
import os
import time

import requests

def test_serper_proxy(endpoint_url: str, query: str, num: int = 10, use_crawl: bool = False, think_content: str = "", web_search_query: str = "", summary_type: str = "once", summary_prompt_type = "webthinker_with_goal"):
    """向Serper代理发送请求并打印响应。"""
    api_url = os.environ.get("SUMMARY_API_URLS").split("|")[0]
    api_key = os.environ.get("SUMMARY_API_KEYS").split("|")[0]
    model = os.environ.get("SUMMARY_MODEL")
    serper_key = os.environ.get("SERPER_API_KEY")
    jina_key = os.environ.get("JINA_API_KEY")

    headers = {
        "Content-Type": "application/json"
        # "X-API-KEY": '60291a6dbc05eee1e409f5fb3aa639059c183eaf|60291a6dbc05eee1e409f5fb3aa639059c1831111'
    }
    payload = {
        "q": query, # 
        "num": num,
        # "use_crawl": use_crawl,
        # "think_content": think_content,
        # "web_search_query": web_search_query,
        # "summary_type": summary_type,
        # "summary_prompt_type": summary_prompt_type,
        # # 从环境变量中获取
        # "api_url": api_url,
        # "api_key": api_key,
        # "jina_key": jina_key,
        # "model": model,
    }

    print(f"--- Sending request for query: '{query}' ---")
    start_time = time.time()
    url = endpoint_url
    # print(url, headers, payload)
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 如果状态码不是 2xx，则引发异常

        elapsed_time = time.time() - start_time
        print(f"Status Code: {response.status_code} (Response time: {elapsed_time:.2f}s)")
        
        # 打印结果摘要
        print(response)
        result = response.json()
        print(f"Results: {result}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serper proxy v3 test script.')
    parser.add_argument('endpoint_url', type=str, help='The endpoint URL to test against (e.g., http://127.0.0.1:9002/search)')
    args = parser.parse_args()
    # # 运行一些测试用例
    print("=== TEST without crawl ===")
    # args.endpoint_url = "127.0.0.1:10001/search"
    test_serper_proxy(args.endpoint_url, 'IPL matches before 2020 where team won by 6 wickets and chased under 19 overs.', num=3)

    # print("=== TEST with crawl ===")
    # test_serper_proxy(args.endpoint_url, "who is the founder of deepmind", 2, use_crawl=True, think_content="I want to know who is the founder of deepmind", web_search_query="who is the founder of deepmind", summary_prompt_type = "webthinker_with_goal")
