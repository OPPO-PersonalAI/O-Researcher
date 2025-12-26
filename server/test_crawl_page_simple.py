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

from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='Crawl page test script.')
parser.add_argument('endpoint_url', type=str, help='The endpoint URL to test against (e.g., http://127.0.0.1:20002/crawl_page)')

# Parse command line arguments
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
        # Send request
        url = args.endpoint_url
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        # Check HTTP status code
        response.raise_for_status()
        
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response content: {response.text}")
            continue

        # Output result
        if result.get("success"):
            print("Success!")
            print(f"Processing time: {result.get('processing_time'):.1f}s")
            print("\nResult:")
            print("-" * 50)
            print(result.get('obs'))
            print("-" * 50)
        else:
            print(f"Failed: {result.get('error_message', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print("Connection error: Cannot connect to server, please ensure server is running")
        break
    except requests.exceptions.Timeout:
        print("Timeout error: Request timed out")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
    except Exception as e:
        print(f"Unknown error: {str(e)}")

print("\nAll tests completed")