#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import time

import requests

def test_serper_proxy(endpoint_url: str, query: str, num: int = 10, use_crawl: bool = False, think_content: str = "", web_search_query: str = "", summary_type: str = "once", summary_prompt_type = "webthinker_with_goal"):
    api_url = os.environ.get("SUMMARY_API_URLS").split("|")[0]
    api_key = os.environ.get("SUMMARY_API_KEYS").split("|")[0]
    model = os.environ.get("SUMMARY_MODEL")
    serper_key = os.environ.get("SERPER_API_KEY")
    jina_key = os.environ.get("JINA_API_KEY")

    headers = {
        "Content-Type": "application/json"
        # "X-API-KEY": 'xxxxxx'
    }
    payload = {
        "q": query,
        "num": num
    }

    print(f"--- Sending request for query: '{query}' ---")
    start_time = time.time()
    url = endpoint_url
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        print(f"Status Code: {response.status_code} (Response time: {elapsed_time:.2f}s)")
        
        print(response)
        result = response.json()
        print(f"Results: {result}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serper proxy v3 test script.')
    parser.add_argument('endpoint_url', type=str, help='The endpoint URL to test against (e.g., http://127.0.0.1:20001/search)')
    args = parser.parse_args()
    print("=== TEST without crawl ===")
    test_serper_proxy(args.endpoint_url, 'IPL matches before 2020 where team won by 6 wickets and chased under 19 overs.', num=3)