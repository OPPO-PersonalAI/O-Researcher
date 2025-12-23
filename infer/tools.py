import os
import random
import requests
from utils import extract_last_tag

# ##################################################################################################################
def WebSearchTool(web_search_url, task, query, history, topk=10):
    """向Serper代理发送请求并打印响应。"""
    # think_content
    think_content = extract_last_tag(history, '<think>', '</think>')
    # web_search_query
    web_search_query = extract_last_tag(history, '<web_search>', '</web_search>')

    if topk > 20:
        topk = 20

    payload = {
        "q": query,
        "num": topk,
        # "use_crawl": False,
        # "task": task,
        # "web_search_query": web_search_query,
        # "think_content": think_content,
        # "api_url": selected_url_key_group["api_url"],
        # "api_key": selected_url_key_group["api_key"],
        # "model": model,
        # "summary_type": "page",
        # "chunk_size": 8192,
        # "do_last_summary": False,
    }

    try:
        response = requests.post(web_search_url, json=payload)
        response.raise_for_status()  # 如果状态码不是 2xx，则引发异常
        result = response.json()

    except requests.exceptions.RequestException as e:
        result = f"An error occurred: {e}"
    return result


def CrawlPageTool(crawl_page_url, task, urls, history, save_only_one_url):
    if isinstance(urls, str):
        urls = urls.split("|")
    # think_content
    think_content = extract_last_tag(history, '<think>', '</think>')
    # web_search_query
    web_search_query = extract_last_tag(history, '<web_search>', '</web_search>')
    data = {
        "urls": urls,
        "task": task, 
        "web_search_query": web_search_query,
        "think_content": think_content,
        "chunk_size": 8192
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(crawl_page_url, json=data, timeout=1000, headers=headers)
    result = response.json()
    if result.get("success"):
        crawl_page_result = result["obs"]
    else:
        crawl_page_result = result.get("error_message")
    return crawl_page_result