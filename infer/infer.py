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

# Create configurations
URL_CONFIG = {
    "config": [MODEL_URL],
    "pointer": 0,
}

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
    parser.add_argument("--retry_attempt", type=int, default=100,
                        help="Maximum retry attempts")

    # Parallel Processing
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel processes")

    # Round Configuration
    parser.add_argument("--round", type=int, default=1,
                        help="Total number of inference rounds")

    # Tool Mode
    parser.add_argument("--remote_tool", type=bool, default=True,
                        help="Use remote tool services")
    parser.add_argument("--save_only_one_url", type=bool, default=False,
                        help="Save only one URL per search")

    # Data Configuration
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON/JSONL file path")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--q_key", type=str, default="question",
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
        "retry_attempt": args.retry_attempt,
        "parallel": args.parallel,
        "save_only_one_url": args.save_only_one_url,
        "web_search_config": web_search_config,
        "crawl_page_config": crawl_page_config,
        "remote_tool": args.remote_tool,
        "round": args.round,
    }

    return args, infer_kwargs


def get_search_results_with_format(task, response, history, **kwargs):
    # 声明全局变量
    global summary_input, summary_output, web_search_config, crawl_page_config
    global wiki_search_config, summary_model_config
    
    try:
        _, tool, parsed_content = extract_specific_tag(response)
        search_results = ""
        remote_tool = kwargs.get("remote_tool", True)
        
        # 获取配置信息
        cur_web_search_config = web_search_config["config"][
            web_search_config["pointer"] % len(web_search_config["config"])]
        cur_crawl_page_config = crawl_page_config["config"][
            crawl_page_config["pointer"] % len(crawl_page_config["config"])]

        # 更新配置指针
        web_search_config["pointer"] += 1
        crawl_page_config["pointer"] += 1

        if tool == '</web_search>':
            # 处理web搜索，支持多个查询
            queries = parsed_content['queries']
            num = parsed_content['num']

            web_results = RemoteWebSearchTool(
                cur_web_search_config[0],
                task=task,
                query="|".join(queries),  # 合并多个查询
                history=history,
                topk=num  # 使用提取的num参数
            )
            search_results = web_results

        elif tool == "</crawl_page>":
            # 处理页面爬取，支持多个URL
            urls = parsed_content['urls']
            crawl_results = RemoteCrawlPageTool(
                crawl_page_url=cur_crawl_page_config[0],
                task=task,
                urls=urls,  # 传递URL列表
                history=history,
                save_only_one_url=kwargs.get("save_only_one_url", "False")
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
                # {"role": "system", "content": "You are a helpful assistant."},
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
    
    retry_attempts = kwargs.get("retry_attempt", 100)
    result_list = []
    attempt = 0
    error_count = 0

    while attempt < retry_attempts and error_count < 10:
        step_list = [elem["type"] for elem in result_list]
        # 检查是否有连续15个重复步骤
        if len(result_list) >= 15:
            for step in range(14, len(result_list)):
                if len(set(item["type"] for item in result_list[-15:]))==1:
                    return result_list, f"special_bad_case: 连续15次重复步骤: {step_list}"
        logging.info(f"开始调用模型: {MODEL}")
        time.sleep(random.random() * 0.1)

        item_type, content = api_client(system_prompt, query, current_answer, fixed_url, KEY, MODEL, **kwargs)
        logging.info(f"调用模型完毕: {MODEL}")
        content_wo_think = content.split("</think>")[-1].strip()
        logging.info(f"step {attempt+1}: {item_type}")
        # if item_type == "error" or " budget " in content.lower() or " 404 " in content.lower() or " 402 " in content.lower():
        if item_type == "error" or content is None:
            error_count += 1
            continue
        elif content_wo_think in "".join(current_answer):
            content = f"|<BEGIN_OF_DUPLICATE_CONTENT>|{content}|<END_OF_DUPLICATE_CONTENT>|You have previsouly output the same content. Please try to proceed further and think differently with no more duplications."
            logging.info(f"found duplicate step: {item_type} | {content_wo_think}")
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
        elif item_type == "answer":
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
            return result_list, None
        elif item_type in ["plan", "reflection", "summary", "double_check"]:
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
        elif item_type == "suggested_answer":
            result_list.append({
                "type": item_type,
                "content": content
            })
            current_answer += content
            attempt += 1
            return current_answer, result_list, None
        elif item_type in ["web_search", "wiki_search", "crawl_page"]:
            logging.info(f"开始调用工具: {item_type}")
            current_answer_with_current_content = current_answer + content
            is_success, obs = get_search_results_with_format(query, content, current_answer_with_current_content, **kwargs)
            logging.info(f"工具调用完毕: {item_type}")
            # if " budget " in obs.lower() or " 404 " in obs.lower() or " 402 " in obs.lower() or "An error occurred:" in obs.lower():
            if not is_success or obs is None or "BudgetExceededError" in obs or "An error occurred:" in obs or "400 Bad Request" in obs:
                logging.warning(f"使用工具{item_type}时发生了调用错误: {obs}")
                error_count += 1
                continue
            result_list.append({
                "type": item_type,
                "content": f"\n\n{content.strip()}\n\n<observation>\n{obs.strip()}\n</observation>"
            })
            current_answer += f"\n\n{content.strip()}\n\n<observation>\n{obs.strip()}\n</observation>"
            attempt += 1
    # 直接用suggested_answer作为answer
    suggested_answer_list = [item for item in result_list if item["type"] == "suggested_answer"]
    if suggested_answer_list:
        result_list.append({
            "type": "suggested_answer",
            "content": suggested_answer_list[-1]["content"]
        })
        return current_answer, result_list, None
    else:
        return current_answer, result_list, "超出max_attempts次数，并且没有找到answer/suggested_answer"


# 提取最后的答案
def extract_prediction_final_simple(content: str) -> str:
    """
    严格按照顺序，依次尝试两种方法来提取答案。
    1. 优先匹配完美的 <suggested_answer>xxx</suggested_answer> 格式。
    2. 如果失败，则尝试匹配以 </suggested_answer> 结尾的格式。
    3. 如果都失败，则抛出 ValueError。
    """
    # Plan A: 尝试完美匹配
    perfect_match = re.search(r'<suggested_answer>(.*?)</suggested_answer>', content, re.DOTALL)
    if perfect_match:
        return perfect_match.group(1).strip()

    # --- Plan B: 如果 Plan A 失败，执行您指定的简单后备方案 ---
    end_tag = "</suggested_answer>"
    if end_tag in content:
        # 直接以结束标签为界，获取它之前的所有内容
        potential_answer = content.split(end_tag)[0]
        cleaned_answer = potential_answer.strip()
        
        # 只要清理后不是空字符串，就返回
        if cleaned_answer:
            return cleaned_answer

    # --- Plan C: 如果 Plan A 和 Plan B 都失败了 ---
    raise ValueError("无法在内容中找到 </suggested_answer> 标签。")


def process_queries(infile, outfile, q_key, a_key, **kwargs):
    # 读取输入数据
    if infile.endswith(".json"):
        questions_data = read_json(infile)
    elif infile.endswith(".jsonl"):
        questions_data = read_jsonl(infile)
    else:
        raise ValueError(f"不支持的文件格式: {infile}")

    # 检查输出文件是否存在并去重
    out_data = []
    out_set = set()
    if os.path.exists(outfile):
        out_data = read_jsonl(outfile)
        out_data = [item for item in out_data if item["article"] is not None]
        write_jsonl(out_data, outfile)
        out_set = set([item[q_key] for item in out_data])

    logging.info(outfile)
    new_questions_data = [item for item in questions_data if item[q_key] not in out_set]
    logging.info(f"初始数据: {len(questions_data)}, 过滤后数据：{len(new_questions_data)}")
    questions_data = new_questions_data

    # 初始化统计信息和共享队列
    stats = {"total": len(new_questions_data), "success": 0, "failed": 0}
    task_queue = Queue()
    result_queue = Queue()
    write_lock = Lock()

    # 生产者函数 - 将任务放入队列
    def producer():
        for idx, question_data in enumerate(questions_data):
            task_queue.put((idx, question_data))
        # 放入结束标记
        for _ in range(kwargs.get("parallel", 4)):
            task_queue.put(None)

    # 消费者函数 - 从队列获取任务并处理
    def consumer():
        fixed_url = URL_CONFIG["config"][URL_CONFIG["pointer"] % len(URL_CONFIG["config"])]
        URL_CONFIG["pointer"] += 1

        nonlocal stats
        while True:
            task = task_queue.get()
            if task is None:
                break

            idx, question_data = task
            question = question_data[q_key]
            level = question_data.get('Level', '-1')

            trace = {
                "question_id": str(idx),
                "question": question,
                "Level": level,
                "prediction": None,
                "llm_judge": 0,
                "steps": [],
                "status": None,
                "error": None,
                "elapsed_time_seconds": None,  # 单条推理耗时（秒）
                "elapsed_time_formatted": None,  # 格式化耗时
            }

            current_answer, result_list, failed_reason = process_single_data(question, fixed_url = fixed_url, **kwargs)

            # 先设置为None
            clean_data = None

            if failed_reason:
                trace["status"] = "error"
                trace["error"] = failed_reason
            elif "BudgetExceededError" in str(result_list):
                trace["status"] = "error"
                trace["error"] = "检测到特殊词汇，属于工具调用问题"
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
                            "elapsed_time_seconds": trace["elapsed_time_seconds"],  # 耗时（秒）
                            "elapsed_time_formatted": trace["elapsed_time_formatted"],  # 格式化耗时
                        }
                        trace["prediction"] = clean_data
                    except Exception as e:
                        print(f"get error : {e}")
            
            result_queue.put((clean_data, trace))
            task_queue.task_done()
    
    # 结果写入函数
    def result_writer():
        nonlocal stats
        while True:
            result_item = result_queue.get()
            if result_item is None:  # 结束标记
                break
            
            result, trace = result_item
            # 只有当 prediction 不为 None 时才处理
            if trace.get("prediction") is not None:
                if result is not None:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                
                # 写入文件
                with write_lock:
                    write_jsonl([result], outfile, "a")
            else:
                # prediction 为 None 的情况不计入统计
                logging.info(f"跳过写入: prediction 为 None (问题: {trace.get('question')})")
            
            result_queue.task_done()

    # 创建线程池
    num_workers = kwargs.get("parallel", 4)
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        # 启动生产者线程
        executor.submit(producer)
        
        # 启动消费者线程
        consumer_futures = [executor.submit(consumer) for _ in range(num_workers)]
        
        # 启动结果写入线程
        writer_future = executor.submit(result_writer)
        
        # 等待所有任务完成
        for future in as_completed(consumer_futures):
            future.result()
        
        # 所有消费者完成后，发送结束信号给写入线程
        result_queue.put(None)
        writer_future.result()

    logging.info(f"处理完成! 成功: {stats['success']}, 失败: {stats['failed']}, 总计: {len(new_questions_data)}")
    return outfile, stats

def main():
    # Parse command line arguments and create configuration
    args, INFER_KWARGS = parse_args_and_create_config()

    # Create show kwargs for display
    SHOW_KWARGS = {
        "key": KEY,
        "model": MODEL,
        "url_config": URL_CONFIG,
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
            # 多轮时，为每轮生成不同的输出文件名
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
    
    # 汇总统计
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
    # cur_web_search_config = "http://10.77.227.97:20001/search"
    # task = '收集整理目前中国9阶层实际收入和财务状况，特别研究得出中国的中产有哪些特点，实际中产人数，财力等等'
    # queries = ['CFPS 家庭金融 资产 负债 调查 ', ' 中国居民家庭收支调查 资产 负债 NBS ', ' China Household Finance Survey 资产 负债 ', ' China Household Income Project 财务 资产 ', ' PBoC 住户部门 资产负债表 报告']
    # history = """<subtask_list>
    # - 在中国9阶层实际收入和财务状况框架下，哪些指标、调查数据源与方法可可靠界定中产的财务结构与资产构成（含房产、金融资产、债务、家庭规模）？
    # - 中国的中产阶层在中国9阶层实际收入和财务状况体系中呈现出哪些行为与消费模式特点（如负债、投资、教育、养老准备、地域差异）？
    # - 在中国9阶层实际收入和财务状况的背景下，中产阶层的人数规模、地域分布与变动趋势是什么（基于人口比例、地区、城市化与家庭规模的差异）？
    # - 基于中国9阶层实际收入和财务状况的系统划分，中产阶层的收入范围、财务指标与典型画像有哪些（含税前/税后收入、家庭可支配收入、支出结构）？
    # - 目前中国的9阶层实际收入与财务状况如何定义与划分，这一框架的来源、指标与应用范围是什么？
    # </subtask_list>

    # </subtask_list>

    # <subtask>
    # 在中国9阶层实际收入和财务状况框架下，哪些指标、调查数据源与方法可可靠界定中产的财务结构与资产构成（含房产、金融资产、债务、家庭规模）？
    # </subtask>

    # <think>
    # We need a concise, high-level plan to define indicators, identify reliable survey data sources, and choose methods to delineate middle-class financial structure under the 9-tier framework in China. The plan should balance theoretical clarity (asset structure components), data quality (nationally representative, longitudinal, harmonized), and methodological rigor (measurement, imputation, classification). We'll leverage web search to locate authoritative datasets and papers (China Family Panel Studies, CFPS; China Household Finance Survey, CHFS; CHIP; Urban and Rural Household Income and Expenditure Surveys; PBoC Financial Stability Reports; CEIC/NBS; GFSA/SAFE for FX debt). Given tool constraints, we prioritize core queries in batches of at most two keywords per search, then crawl primary dataset pages. We will synthesize indicators and methods, map them to the 9-tier income bands, and outline validation and robustness checks. Deliverables include a reference-backed indicator set, data dictionary, codebook, and reproducible code.
    # </think>

    # <plan>
    # High-level plan:

    # 1) Clarify scope and 9-tier framework
    # - Define the 9-tier income classification in China (e.g., decile-based or income band thresholds) and the middle-class target (e.g., 4th–6th deciles). Confirm if urban/rural or regional adjustments are needed.
    # - Enumerate required financial structure dimensions: housing (owned, mortgaged, value), financial assets (cash, deposits, bank wealth, mutual funds, stocks, bonds, insurance, trusts), non-financial assets (vehicles, durable goods), debt (mortgage, consumer, education, informal), and family structure (size, lifecycle stage, hukou).

    # 2) Build the indicator dictionary
    # - Income and classification: household disposable income, per capita income, equivalized income (OECD-modified), income volatility.
    # - Asset structure: total net worth, housing wealth share, financial wealth share, liquid assets share, risky asset share, wealth-to-income ratio.
    # - Debt structure: total liabilities, mortgage balance and LTV, consumer debt (credit cards, personal loans), debt-to-income, debt-service-to-income, financial leverage (liabilities/net worth).
    # - Family context: household size, presence of children/elders, urban vs rural, region, hukou status, lifecycle stage (age of head), education.

    # 3) Identify and vet data sources
    # - Priority national surveys with asset/debt modules: CFPS (China Family Panel Studies), CHFS (China Household Finance Survey), CHIP (China Household Income Project), Urban/Rural Household Income and Expenditure Surveys, CGSS with wealth modules.
    # - Administrative/aggregate sources for validation: NBS Statistical Yearbooks, PBoC Financial Stability Report (household balance sheet), CEIC, SAFE/GFSA (external debt), regulators’ bank loan data (mortgage volume), MOHURD/real estate yearbooks for housing prices.
    # - Foreign comparators for methodological guidance: SEDS/SCF (US), HFCS (Euro), SHIW (Italy), HSE (UK).
    # - Use web_search in batches (max two keywords per query) to locate official dataset pages, documentation, and key papers on middle-class measurement in China; then use crawl_page on 3–5 primary URLs (CFPS, CHFS, CHIP, NBS, PBoC).

    # 4) Data harmonization and preprocessing
    # - Select a primary dataset (e.g., CFPS for broad coverage, CHFS for detailed balance sheets), with secondary datasets for cross-validation.
    # - Map variables to the indicator dictionary; construct derived metrics (e.g., equivalized income, net worth, debt ratios).
    # - Handle missing values via multiple imputation (e.g., MICE), and document imputation models per module (income, assets, debt).
    # - Address underreporting of assets/income using calibration (e.g., raking to external totals from NBS/PBoC) and top-coding adjustment; document sensitivity.

    # 5) Classify middle class within the 9-tier framework
    # - Assign households to 9 tiers based on equivalized disposable income within urban/rural and regional strata; consider harmonized CPI adjustments for purchasing power.
    # - Define middle class as a contiguous set of tiers (e.g., 4th–6th deciles) and report sensitivity to alternative definitions (tertiles, thresholds vs percentiles).

    # 6) Measure financial structure and asset composition
    # - Construct distributions of asset/debt indicators by tier: mean, median, interdecile ranges, Lorenz curves, wealth-to-income elasticities.
    # - Decompose net worth by component (housing, financial, non-financial) and by risk class (liquid vs illiquid; risky vs safe assets).
    # - Examine heterogeneity by family size, lifecycle, hukou, region; report elasticities and partial correlations.

    # 7) Methodological robustness checks
    # - Compare income tiers across CFPS vs CHFS; reconcile discrepancies via reweighting and variable mapping.
    # - Sensitivity to income concept (market vs disposable), equivalence scales, and price deflators.
    # - Validate asset aggregates against PBoC/NBS totals; ensure mortgage balances align with banking statistics.
    # - Triangulate debt metrics with PBoC household debt/GDP and loan composition data.

    # 8) Deliverables and reproducibility
    # - Produce: indicator dictionary; data dictionary/codebook; sample classification rules for 9 tiers; replication code (version-controlled) for cleaning, imputation, classification, and analysis; reference-backed report with citations.
    # - Document assumptions, definitions, and limitations; include a quality scorecard per data source (coverage, asset/debt granularity, frequency, sampling).

    # 9) Timeline and roles
    # - Week 1: Framework definition, indicator dictionary, search and source vetting.
    # - Week 2: Data acquisition (public extracts), preprocessing, imputation models.
    # - Week 3: Classification and measurement, robustness checks, validation against external aggregates.
    # - Week 4: Synthesis, write-up with references, deliverables packaging.
    # - Roles: Lead economist (framework/interpretation), Data engineer (ETL/harmonization), Quant analyst (imputation/methods), Research assistant (search/citations).

    # 10) Risk management
    # - If key modules are missing (e.g., CHFS not publicly available), pivot to CFPS with asset imputation using donor matching and microsimulation, with clear disclosure.
    # - Ensure ethical compliance and data use restrictions; use aggregate validation only where necessary to protect microdata integrity.
    # </plan>

    # <think>
    # Task analysis: The user asks, in Chinese, for indicators, survey data sources, and methods that reliably delineate middle-class financial structure and asset composition (including housing, financial assets, debt, and family size) within China's 9-tier income/wealth framework. This requires identifying authoritative datasets (e.g., CFPS, CHFS, CHIP), methodological papers on middle-class measurement in China, and guidance on equivalence scales and debt/income metrics, as well as administrative aggregates (PBoC, NBS) for validation. Tool matching: Among the available tools, web_search is appropriate to discover and retrieve candidate sources; crawl_page can be used subsequently to extract details from selected URLs. suggested_answer should not be used immediately, as the instruction explicitly asks not to choose it after the plan and to first perform analysis using tools. Parameter check: web_search requires 'query' (3–5 orthogonal queries separated by '|') and 'num' (integer). I will craft 5 orthogonal Chinese queries covering datasets (CFPS/CHFS/CHIP), middle-class measurement frameworks, and validation sources (PBoC/NBS). I will set num to 10 to retrieve sufficient breadth without redundancy. This avoids repeating any prior identical parameters since no previous tool call was made.
    # </think>

    # <web_search>query=CFPS 家庭金融 资产 负债 调查 | 中国居民家庭收支调查 资产 负债 NBS | China Household Finance Survey 资产 负债 | China Household Income Project 财务 资产 | PBoC 住户部门 资产负债表 报告&num=10</web_search>"""

    # num = 10

    # web_results = RemoteWebSearchTool(
    #                 cur_web_search_config,
    #                 task=task,
    #                 query="|".join(queries),  # 合并多个查询
    #                 history=history,
    #                 topk=num  # 使用提取的num参数
    #             )

    # print(web_results)

    main()



# if __name__ == "__main__":
#     for key, value in SHOW_KWARGS.items():
#         logging.info(f">>>> {key}: {value}")

#     infile="/home/notebook/code/group/eason/AFM_module/deep_research_bench/data/prompt_data/query_five.jsonl"
#     outfile = f"/home/notebook/code/group/eason/AFM_module/infer/afm_infer_data/1205_swift_72B_zh_sft_80k_rl_step3_five_{MODEL}.summary_model_{summary_model_config['model_id']}.remote_tool_{INFER_KWARGS['remote_tool']}.jsonl"

#     q_key = "prompt"
#     a_key = "answer"

#     # 设置并行线程数
#     start_time = time.time()
#     all_outfiles = []
#     for current_round in range(INFER_KWARGS["round"]):
#         current_outfile = outfile.replace(".jsonl", f".round_{current_round}.jsonl")
#         all_outfiles.append(current_outfile)
#         process_queries(infile, current_outfile, q_key, a_key, **INFER_KWARGS)
#     cost_time = time.time() - start_time
#     print(f"时间花费: {cost_time}")