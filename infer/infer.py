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

import json
import re
import time
import random
import os
random.seed(1234)
import argparse
from openai import OpenAI
from queue import Queue
import logging
from threading import Lock
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

from tools import (
    WebSearchTool as RemoteWebSearchTool,
    CrawlPageTool as RemoteCrawlPageTool
)

from utils import (
    read_jsonl,
    write_jsonl,
    read_json,
    write_json,
    count_tokens,
    extract_last_tag,
    extract_specific_tag
)

from prompts import sys_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def get_required_env(key):
    """Get required environment variable, raise error if not found."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable '{key}' is not set. Please set it before running the script.")
    return value

# Load required environment variables
MODEL = get_required_env("MODEL_NAME")
MODEL_PATH=get_required_env("MODEL_PATH")
MODEL_URL = get_required_env("MODEL_URL")
WEBSEARCH_URL = get_required_env("WEBSEARCH_URL")
CRAWL_PAGE_URL = get_required_env("CRAWL_PAGE_URL")

# Create configurations - support multiple URLs separated by |
url_list = [url.strip() for url in MODEL_URL.split("|") if url.strip()]
URL_CONFIG = {
    "config": url_list,
    "pointer": 0,
    "lock": Lock(),  # Thread-safe round-robin
}

def get_next_url():
    """Get next URL using round-robin (thread-safe)"""
    with URL_CONFIG["lock"]:
        url = URL_CONFIG["config"][URL_CONFIG["pointer"] % len(URL_CONFIG["config"])]
        URL_CONFIG["pointer"] += 1
        return url

KEY = "empty"
SYSTEM_PROMPT = sys_prompt

## External API ##
## Tool Server ##
### web_search ###
web_search_config = {
    "config": [
        [WEBSEARCH_URL],
    ],
    "pointer": 0,
}

### crawl_page ###
crawl_page_config = {
    "config": [
        [CRAWL_PAGE_URL],
    ],
    "pointer": 0,
}


try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception as e:
    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizer_file",
        trust_remote_code=True)

def parse_args_and_create_config():
    """Parse command line arguments and create inference configuration for A²FM."""
    parser = argparse.ArgumentParser(description="A²FM: Adaptive Agent Foundation Model Inference")

    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0.0 to 2.0)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter (0.0 to 1.0)")
    parser.add_argument("--presence_penalty", type=float, default=0,
                        help="Presence penalty (-2.0 to 2.0)")
    parser.add_argument("--frequency_penalty", type=float, default=0,
                        help="Frequency penalty (-2.0 to 2.0)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum tokens per generation")
    parser.add_argument("--total_tokens", type=int, default=81920,
                        help="Maximum total tokens for generation")

    # Tool Configuration
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum inference steps per question")

    # Parallel Processing
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel processes")

    # Round Configuration
    parser.add_argument("--round", type=int, default=1,
                        help="Total number of inference rounds")

    # Data Configuration
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON/JSONL file path")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--q_key", type=str, default="prompt",
                        help="Key name for question field in input data")
    parser.add_argument("--a_key", type=str, default="answer",
                        help="Key name for answer field in input data")

    # Parse arguments
    args = parser.parse_args()

    # Create inference kwargs from arguments
    infer_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "max_tokens": args.max_tokens,
        "total_tokens": args.total_tokens,
        "max_steps": args.max_steps,
        "parallel": args.parallel,
        "web_search_config": web_search_config,
        "crawl_page_config": crawl_page_config,
        "round": args.round,
    }

    return args, infer_kwargs


def get_search_results_with_format(task, response, history, **kwargs):
    global summary_input, summary_output, web_search_config, crawl_page_config
    global wiki_search_config, summary_model_config
    
    try:
        _, tool, parsed_content = extract_specific_tag(response)
        search_results = ""
        remote_tool = kwargs.get("remote_tool", True)
        
        cur_web_search_config = web_search_config["config"][
            web_search_config["pointer"] % len(web_search_config["config"])]
        cur_crawl_page_config = crawl_page_config["config"][
            crawl_page_config["pointer"] % len(crawl_page_config["config"])]

        web_search_config["pointer"] += 1
        crawl_page_config["pointer"] += 1

        if tool == '</web_search>':
            # Handle web search, supports multiple queries
            queries = parsed_content['queries']
            num = parsed_content['num']

            web_results = RemoteWebSearchTool(
                cur_web_search_config[0],
                task=task,
                query="|".join(queries),
                history=history,
                topk=num
            )
            search_results = web_results

        elif tool == "</crawl_page>":
            # Handle page crawling, supports multiple URLs
            urls = parsed_content['urls']
            crawl_results = RemoteCrawlPageTool(
                crawl_page_url=cur_crawl_page_config[0],
                task=task,
                urls=urls,
                history=history
            )
            search_results = crawl_results

        else:
            search_results = ''
            
        return True, search_results
    except Exception as err:
        return False, str(err)

def api_client(system, prompt, current_answer, url, key, model, stop_words=None, **kwargs):
    if stop_words is None:
        stop_words = [
            "</web_search>", 
            "</crawl_page>",
            "</suggested_answer>"
        ]

    client = OpenAI(base_url=url, api_key=key)
    try:
        system_token_count, prompt_token_count, current_answer_token_count = count_tokens(system, tokenizer), count_tokens(prompt, tokenizer), count_tokens(current_answer, tokenizer)
        max_tokens_for_answer = kwargs.get("total_tokens", 16384) - kwargs.get("max_tokens", 2048) - system_token_count - prompt_token_count - current_answer_token_count - 512

        model_output_message = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": f"user\n\n{system}\n\n{prompt}"},
                {"role": "assistant", "content": current_answer}
            ],
            stream=False,
            stop=stop_words,
            temperature=kwargs.get("temperature", 1.0), 
            top_p=kwargs.get("top_p", 1.0),        
            presence_penalty=kwargs.get("presence_penalty", 0), 
            frequency_penalty=kwargs.get("frequency_penalty", 0), 
            max_tokens=max_tokens_for_answer,
            n=1, 
        )

        collected_content = []
        model_output = model_output_message.choices[0].message.content
        stop_tag = model_output_message.choices[0].model_extra["stop_reason"]
        tag = stop_tag.lstrip("</").rstrip(">")

        return tag, model_output + stop_tag
    except Exception as e:
        error_msg = str(e)
        return "error", error_msg

def process_single_data(query, fixed_url, **kwargs):
    system_prompt = SYSTEM_PROMPT.strip()
    current_answer = ""
    
    max_steps = kwargs.get("max_steps", 100)
    result_list = []
    step = 0
    error_count = 0

    while step < max_steps and error_count < 10:
        step_list = [elem["type"] for elem in result_list]
        # Check for 15 consecutive duplicate steps
        if len(result_list) >= 15:
            for step in range(14, len(result_list)):
                if len(set(item["type"] for item in result_list[-15:]))==1:
                    return result_list, f"special_bad_case: 15 consecutive duplicate steps: {step_list}"
        logging.info(f"Calling model: {MODEL}")
        time.sleep(random.random() * 0.1)

        item_type, content = api_client(system_prompt, query, current_answer, fixed_url, KEY, MODEL, **kwargs)
        logging.info(f"Model call completed: {MODEL}")
        content_wo_think = content.split("</think>")[-1].strip()
        logging.info(f"Step {step+1}/{max_steps}: {item_type}")

        if item_type == "error" or content is None:
            error_count += 1
            continue
        elif content_wo_think in "".join(current_answer):
            content = f"|<BEGIN_OF_DUPLICATE_CONTENT>|{content}|<END_OF_DUPLICATE_CONTENT>|You have previsouly output the same content. Please try to proceed further and think differently with no more duplications."
            logging.info(f"Found duplicate step: {item_type} | {content_wo_think}")
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            step += 1
        elif item_type == "answer":
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            step += 1
            return result_list, None
        elif item_type in ["plan", "reflection", "summary", "double_check"]:
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            step += 1
        elif item_type == "suggested_answer":
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            step += 1
            return current_answer, result_list, None
        elif item_type in ["web_search", "wiki_search", "crawl_page"]:
            logging.info(f"Calling tool: {item_type}")
            current_answer_with_current_content = current_answer + content
            is_success, obs = get_search_results_with_format(query, content, current_answer_with_current_content, **kwargs)
            logging.info(f"Tool call completed: {item_type}")

            if not is_success or obs is None or "BudgetExceededError" in obs or "An error occurred:" in obs or "400 Bad Request" in obs:
                logging.warning(f"Tool {item_type} call error: {obs}")
                error_count += 1
                continue
            result_list.append({
                "type": item_type,
                "content": f"\n\n{content.strip()}\n\n<observation>\n{obs.strip()}\n</observation>"
            })
            current_answer += f"\n\n{content.strip()}\n\n<observation>\n{obs.strip()}\n</observation>"
            step += 1
    # Use suggested_answer as answer directly
    suggested_answer_list = [item for item in result_list if item["type"] == "suggested_answer"]
    if suggested_answer_list:
        result_list.append({
            "type": "suggested_answer",
            "content": suggested_answer_list[-1]["content"]
        })
        return current_answer, result_list, None
    else:
        return current_answer, result_list, f"Exceeded max steps ({max_steps}), suggested_answer not found"


# Extract final answer
def extract_prediction_final_simple(content: str) -> str:
    """
    Extract answer using two methods in order:
    1. First try to match perfect <suggested_answer>xxx</suggested_answer> format.
    2. If failed, try to match content ending with </suggested_answer>.
    3. If both fail, raise ValueError.
    """
    # Plan A: Try perfect match
    perfect_match = re.search(r'<suggested_answer>(.*?)</suggested_answer>', content, re.DOTALL)
    if perfect_match:
        return perfect_match.group(1).strip()

    # Plan B: If Plan A fails, use simple fallback
    end_tag = "</suggested_answer>"
    if end_tag in content:
        # Get all content before the end tag
        potential_answer = content.split(end_tag)[0]
        cleaned_answer = potential_answer.strip()
        
        # Return if cleaned answer is not empty
        if cleaned_answer:
            return cleaned_answer

    # Plan C: If both Plan A and Plan B fail
    raise ValueError("Cannot find </suggested_answer> tag in content.")


def process_queries(infile, outfile, q_key, a_key, **kwargs):
    if infile.endswith(".json"):
        questions_data = read_json(infile)
    elif infile.endswith(".jsonl"):
        questions_data = read_jsonl(infile)
    else:
        raise ValueError(f"Unsupported file format: {infile}")

    out_data = []
    out_set = set()
    if os.path.exists(outfile):
        out_data = read_jsonl(outfile)
        out_data = [item for item in out_data if item["article"] is not None]
        write_jsonl(out_data, outfile)
        out_set = set([item[q_key] for item in out_data])

    logging.info(outfile)
    new_questions_data = [item for item in questions_data if item[q_key] not in out_set]
    logging.info(f"Initial data: {len(questions_data)}, After filtering: {len(new_questions_data)}")
    questions_data = new_questions_data

    stats = {"total": len(new_questions_data), "success": 0, "failed": 0}
    task_queue = Queue()
    result_queue = Queue()
    write_lock = Lock()

    # Producer function - put tasks into queue
    def producer():
        for idx, question_data in enumerate(questions_data):
            task_queue.put((idx, question_data))
        for _ in range(kwargs.get("parallel", 4)):
            task_queue.put(None)

    # Consumer function - get tasks from queue and process
    def consumer():
        nonlocal stats
        while True:
            task = task_queue.get()
            if task is None:
                break

            idx, question_data = task
            question = question_data[q_key]
            level = question_data.get('Level', '-1')
            
            # Round-robin select URL for each query (same URL for all steps of a query to maintain kv_cache)
            fixed_url = get_next_url()

            trace = {
                "question_id": str(idx),
                "question": question,
                "Level": level,
                "prediction": None,
                "llm_judge": 0,
                "steps": [],
                "status": None,
                "error": None,
                "elapsed_time_seconds": None,
                "elapsed_time_formatted": None,
                "model_url": fixed_url,
            }

            current_answer, result_list, failed_reason = process_single_data(question, fixed_url=fixed_url, **kwargs)

            clean_data = None

            if failed_reason:
                trace["status"] = "error"
                trace["error"] = failed_reason
            elif "BudgetExceededError" in str(result_list):
                trace["status"] = "error"
                trace["error"] = "Detected special keywords, tool call issue"
            else:
                trace["steps"] = result_list
                if result_list[-1]["type"] == "suggested_answer":
                    try:
                        prediction = extract_prediction_final_simple(result_list[-1]["content"])

                        clean_data = {
                            "id": task[1]["id"],
                            "prompt": task[1]["prompt"],
                            "article": prediction,
                            "current_answer": current_answer,
                            "elapsed_time_seconds": trace["elapsed_time_seconds"],
                            "elapsed_time_formatted": trace["elapsed_time_formatted"],
                        }
                        trace["prediction"] = clean_data
                    except Exception as e:
                        print(f"get error : {e}")
            
            result_queue.put((clean_data, trace))
            task_queue.task_done()
    
    # Result writer function
    def result_writer():
        nonlocal stats
        while True:
            result_item = result_queue.get()
            if result_item is None:
                break
            
            result, trace = result_item

            if trace.get("prediction") is not None:
                if result is not None:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                
                with write_lock:
                    write_jsonl([result], outfile, "a")
            else:
                logging.info(f"Skip writing: prediction is None (question: {trace.get('question')})")
            
            result_queue.task_done()

    # Create thread pool
    num_workers = kwargs.get("parallel", 4)
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        executor.submit(producer)
        
        consumer_futures = [executor.submit(consumer) for _ in range(num_workers)]

        writer_future = executor.submit(result_writer)

        for future in as_completed(consumer_futures):
            future.result()
        
        result_queue.put(None)
        writer_future.result()

    logging.info(f"Processing completed! Success: {stats['success']}, Failed: {stats['failed']}, Total: {len(new_questions_data)}")
    return outfile, stats

def main():
    # Parse command line arguments and create configuration
    args, INFER_KWARGS = parse_args_and_create_config()

    # Create show kwargs for display (use url list only, Lock is not serializable)
    SHOW_KWARGS = {
        "key": KEY,
        "model": MODEL,
        "url_config": URL_CONFIG["config"],
        "system_prompt": SYSTEM_PROMPT,
        **INFER_KWARGS
    }

    # Display configuration information
    logging.info("=" * 50)
    logging.info("O-Researcher Inference Configuration")
    logging.info("=" * 50)
    for key, value in SHOW_KWARGS.items():
        if key not in ["web_search_config", "crawl_page_config"]:
            logging.info(f">>>> {key}: {value}")

    logging.info("=" * 50)
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Question key: {args.q_key}")
    logging.info(f"Answer key: {args.a_key}")
    logging.info("=" * 50)

    # Process dataset with multiple rounds
    start_time = time.time()
    total_rounds = INFER_KWARGS.get("round", 1)
    all_stats = []
    all_outfiles = []
    
    for current_round in range(total_rounds):
        if total_rounds > 1:
            current_outfile = args.output_file.replace(".jsonl", f".round_{current_round + 1}.jsonl")
            logging.info(f"\n{'=' * 50}")
            logging.info(f"Round {current_round + 1}/{total_rounds}")
            logging.info(f"Output file: {current_outfile}")
            logging.info(f"{'=' * 50}")
        else:
            current_outfile = args.output_file
        
        all_outfiles.append(current_outfile)
        
        try:
            dataset_path, stats = process_queries(
                args.input_file,
                current_outfile,
                args.q_key,
                args.a_key,
                **INFER_KWARGS,
            )
            all_stats.append({"round": current_round + 1, "outfile": current_outfile, **stats})
        except Exception as e:
            logging.error(f"Error in round {current_round + 1}: {str(e)}")
            all_stats.append({"round": current_round + 1, "outfile": current_outfile, "error": str(e)})

    total_cost_time = time.time() - start_time
    logging.info(f"\n{'=' * 50}")
    logging.info(f"All rounds completed!")
    logging.info(f"Total time: {total_cost_time:.2f} seconds")
    
    if total_rounds > 1:
        logging.info(f"Output files:")
        for outfile in all_outfiles:
            logging.info(f"  - {outfile}")
        
        total_success = sum(s.get("success", 0) for s in all_stats)
        total_failed = sum(s.get("failed", 0) for s in all_stats)
        logging.info(f"Total success: {total_success}, Total failed: {total_failed}")

    # Save statistics
    stats_file = args.output_file.replace(".jsonl", ".param_stats.json")
    final_stats = {
        **SHOW_KWARGS,
        "total_rounds": total_rounds,
        "round_stats": all_stats,
        "output_files": all_outfiles,
        "total_time": total_cost_time
    }
    write_json(final_stats, stats_file)
    logging.info(f"Stats saved to: {stats_file}")

if __name__ == "__main__":
    main()