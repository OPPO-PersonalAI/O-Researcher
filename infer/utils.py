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
import os
import re
import json
import tiktoken
from collections import deque
from typing import List, Dict, Any, Union

def extract_specific_tag(text):
    ALLOWED_TAGS = {'think', 'plan', 'reflection', 'suggested_answer', 'answer', 'wiki_search', 
                   'web_search', 'code_execute', 'crawl_page', 'double_check'}

    tag_stack = deque()
    tag_pairs = []

    # 改进正则表达式，确保匹配所有允许的标签
    split_pattern = re.compile(r'(<\/?(?:{})>)'.format('|'.join(ALLOWED_TAGS)))
    segments = split_pattern.split(text)
    segments = [s for s in segments if s.strip()]

    content_buffer = []
    think_content = ''  # 初始化think_content变量

    for seg in segments:
        if seg.startswith('<'):
            is_close_tag = seg.startswith('</')
            tag_name = seg.strip('<>/').lower()

            if tag_name not in ALLOWED_TAGS:
                content_buffer.append(seg)
                continue

            if not is_close_tag:
                # 处理开始标签
                tag_stack.append((tag_name, len(content_buffer)))
            else:
                # 处理结束标签
                if not tag_stack:
                    continue
                open_tag, content_start_idx = tag_stack.pop()

                if open_tag == tag_name:
                    # 提取标签内内容
                    paired_content = ''.join(content_buffer[content_start_idx:])
                    tag_pairs.append({
                        "tool": open_tag,
                        "content": paired_content.strip()
                    })
                    # 更新内容缓冲区
                    content_buffer = content_buffer[:content_start_idx]
                    
                    # 处理think标签内容
                    if "think" in open_tag:
                        think_content = paired_content.strip()
        else:
            content_buffer.append(seg)

    # 处理标签对为空的情况
    if not tag_pairs:
        return "", "", ""

    # 提取最后一个标签的信息并处理
    last_tag = tag_pairs[-1]
    tool = f"</{last_tag['tool']}>"
    content = last_tag['content']
    
    # 针对不同标签进行内容解析
    if last_tag['tool'] == 'web_search':
        # 提取query和num参数
        query_match = re.search(r'query=(.*?)(&|$)', content)
        if "num=" in content:
            num_match = re.search(r'num=(\d+)', content)
        else:
            num_match = re.search(r'serp_num=(\d+)', content)
        
        queries = query_match.group(1).split('|') if query_match else []
        num = int(num_match.group(1)) if num_match else 10  # 默认值
        
        # 构建结构化内容
        parsed_content = {
            'queries': queries,
            'num': num,
            'raw': content
        }
        return think_content, tool, parsed_content
        
    elif last_tag['tool'] == 'crawl_page':
        # # 提取URL列表
        # url_match = re.search(r'url=(.*)', content)

        # 提取URL列表
        url_match = re.search(r'(.*)', content)
        urls = url_match.group(1).split('|') if url_match else []
        
        # 构建结构化内容
        parsed_content = {
            'urls': urls,
            'raw': content
        }
        return think_content, tool, parsed_content
        
    elif last_tag['tool'] == 'code_execute':
        # 提取代码内容
        code_match = re.search(r'code=(.*)', content, re.DOTALL)
        code = code_match.group(1).strip() if code_match else content
        
        # 构建结构化内容
        parsed_content = {
            'code': code,
            'raw': content
        }
        return think_content, tool, parsed_content

    # 其他标签直接返回原始内容
    return think_content, tool, content

######################################################################################################
def extract_last_tag(text, start_tag, end_tag):
    # 查找最后一个结束标签的位置
    end_index = text.rfind(end_tag)
    if end_index == -1:
        return ""
    # 查找对应的开始标签的位置
    start_index = text.rfind(start_tag, 0, end_index)
    if start_index == -1:
        return ""
    # 提取标签内的内容
    start_index += len(start_tag)
    return text[start_index:end_index]

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    读取JSONL（JSON Lines）文件并返回数据列表
    每行解析为一个JSON对象，合并为列表返回
    
    参数:
        file_path (str): JSONL文件路径
    
    返回:
        List[Dict[str, Any]]: 包含所有JSON对象的列表
    
    异常:
        处理文件不存在和解析错误，返回空列表并打印警告
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # 去除行首尾空白并解析JSON
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行解析失败，跳过该条目")
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
    return data


def write_jsonl(
    data: List[Dict[str, Any]], 
    file_path: str, 
    append: bool = False, 
    ensure_ascii: bool = False
) -> bool:
    """
    将数据列表写入JSONL文件，每行一个JSON对象
    支持追加模式和非ASCII字符处理
    
    参数:
        data (List[Dict[str, Any]]): 要写入的JSON对象列表
        file_path (str): 目标文件路径
        append (bool): 是否追加模式，默认False（覆盖写入）
        ensure_ascii (bool): 是否转义非ASCII字符，默认False（保留中文等字符）
    
    返回:
        bool: 写入是否成功
    
    示例:
        write_jsonl([{"key": "value"}], "data.jsonl")
    """
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as file:
            for item in data:
                # 转换为JSON字符串并确保非ASCII字符正确编码
                json_line = json.dumps(item, ensure_ascii=ensure_ascii) + '\n'
                file.write(json_line)
        return True
    except Exception as e:
        print(f"写入文件时发生错误：{str(e)}")
        return False


def read_json(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    读取标准JSON文件并返回解析后的Python对象
    支持对象格式和数组格式的JSON文件
    
    参数:
        file_path (str): JSON文件路径
    
    返回:
        Union[Dict[str, Any], List[Dict[str, Any]]]: JSON解析后的Python对象
    
    异常:
        处理文件不存在、解析错误等情况，返回None并打印错误信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{str(e)}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None


def write_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    file_path: str, 
    indent: int = 2, 
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> bool:
    """
    将Python对象写入标准JSON文件
    支持字典和列表类型的数据
    
    参数:
        data (Union[Dict[str, Any], List[Dict[str, Any]]]): 要写入的Python对象
        file_path (str): 目标文件路径
        indent (int): 缩进空格数，默认为2（美观格式）
        ensure_ascii (bool): 是否转义非ASCII字符，默认False（保留中文等字符）
        sort_keys (bool): 是否按键排序，默认False
    
    返回:
        bool: 写入是否成功
    
    示例:
        write_json({"key": "value"}, "data.json")
    """
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(
                data, 
                file, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys
            )
        return True
    except Exception as e:
        print(f"写入文件时发生错误：{str(e)}")
        return False


def count_tokens(text, tokenizer):
    """使用transformers分词器计算文本的token数量"""
    if not text:
        return 0
    return len(tokenizer.encode(text))


def truncate_special_tokens(text: str, max_tokens: int, tokenizer: str) -> str:
    """
    按照特殊token划分文本，计算token数并根据最大token数从后往前保留内容，
    只保留<plan>、<|FunctionExecute|>标签及其前面的一个<think>标签，删除其他标签
    
    参数:
    text (str): 包含特殊token的文本
    max_tokens (int): 最大保留的token数
    model (str): 用于tokenization的模型名称，默认为"gpt-4-1106-preview"
    
    返回:
    str: 截断后的文本
    """
    
    # 定义特殊token的正则表达式模式
    token_pattern = r'<[^>]+>|</[^>]+>'
    
    # 按照特殊token分割文本
    segments = re.split(f'({token_pattern})', text)
    segments = [seg for seg in segments if seg]  # 移除空片段
    
    if not segments:
        return ""
    
    # 计算每个片段的token数
    segment_tokens = []
    for segment in segments:
        tokens = tokenizer.encode(segment)
        segment_tokens.append((segment, len(tokens)))
    
    # 找到所有plan和reflection标签的位置
    keep_tags = ['plan', 'reflection']
    keep_tag_indices = []
    
    for i, (segment, _) in enumerate(segment_tokens):
        tag_match = re.match(r'<(/?)([^>]+)>', segment)
        if tag_match:
            is_closing = tag_match.group(1) == '/'
            tag_name = tag_match.group(2)
            if tag_name in keep_tags:
                keep_tag_indices.append((i, tag_name, is_closing))
    
    # 收集需要保留的特殊标签片段索引
    special_indices = set()
    
    # 处理每个特殊标签，确保标签对完整
    for i, tag_name, is_closing in keep_tag_indices:
        if is_closing:
            # 结束标签，查找对应的开始标签
            start_idx = -1
            depth = 0
            for j in range(i, -1, -1):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth += 1
                        else:
                            depth -= 1
                            if depth == 0:
                                start_idx = j
                                break
            if start_idx != -1:
                for idx in range(start_idx, i + 1):
                    special_indices.add(idx)
        else:
            # 开始标签，查找对应的结束标签
            end_idx = -1
            depth = 0
            for j in range(i, len(segments)):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth -= 1
                            if depth == 0:
                                end_idx = j
                                break
                        else:
                            depth += 1
            if end_idx != -1:
                for idx in range(i, end_idx + 1):
                    special_indices.add(idx)
    
    # 计算特殊标签部分的token数
    special_segments = [segment_tokens[i] for i in sorted(special_indices)]
    special_tokens = sum(token_count for _, token_count in special_segments)
    
    # 如果特殊标签部分已经超过token限制，截断并返回
    if special_tokens > max_tokens:
        result = []
        current_tokens = 0
        
        for segment, token_count in reversed(special_segments):
            if current_tokens + token_count <= max_tokens:
                result.insert(0, segment)
                current_tokens += token_count
            else:
                if re.match(r'</[^>]+>', segment):
                    tag_name = re.search(r'</([^>]+)>', segment).group(1)
                    start_tag_found = False
                    print(result)
                    for s, _ in reversed(result):
                        if re.match(rf'<{tag_name}>', s):
                            start_tag_found = True
                            break
                    if not start_tag_found:
                        for s, tc in reversed(special_segments):
                            if re.match(rf'<{tag_name}>', s):
                                if current_tokens + tc <= max_tokens:
                                    result.insert(0, s)
                                    current_tokens += tc
                                break
        return ''.join(result) if result else segments[0]
    
    # 还有剩余token，从后往前添加其他内容
    remaining_tokens = max_tokens - special_tokens
    additional_segments = []
    
    # 从后往前遍历所有片段
    for i in range(len(segments) - 1, -1, -1):
        if i in special_indices:
            continue  # 跳过已经保留的特殊标签
        
        segment, token_count = segment_tokens[i]
        
        # 如果添加当前片段会超出限制，尝试部分添加或跳过
        if remaining_tokens <= 0:
            break
        
        if token_count <= remaining_tokens:
            additional_segments.insert(0, segment)
            remaining_tokens -= token_count
        else:
            # 尝试部分添加文本内容
            if not re.match(r'<[^>]+>|</[^>]+>', segment):  # 普通文本
                tokens = tokenizer.encode(segment)
                partial_tokens = tokens[:remaining_tokens]
                partial_text = tokenizer.decode(partial_tokens)
                if partial_text:
                    additional_segments.insert(0, partial_text)
                    remaining_tokens = 0
    
    # 组合结果：特殊标签部分 + 从后往前添加的其他内容
    return ''.join([seg for seg, _ in special_segments] + additional_segments)



def truncate_special_tokens_64k(text: str, max_tokens: int, tokenizer: str) -> str:
    """
    按照特殊token划分文本，计算token数并根据最大token数从后往前保留内容，
    只保留<plan>、<|FunctionExecute|>标签及其前面的一个<think>标签，删除其他标签
    
    参数:
    text (str): 包含特殊token的文本
    max_tokens (int): 最大保留的token数
    model (str): 用于tokenization的模型名称，默认为"gpt-4-1106-preview"
    
    返回:
    str: 截断后的文本
    """
    
    # 定义特殊token的正则表达式模式
    token_pattern = r'<[^>]+>|</[^>]+>'
    
    # 按照特殊token分割文本
    segments = re.split(f'({token_pattern})', text)
    segments = [seg for seg in segments if seg]  # 移除空片段
    
    if not segments:
        return ""
    
    # 计算每个片段的token数
    segment_tokens = []
    for segment in segments:
        tokens = tokenizer.encode(segment)
        segment_tokens.append((segment, len(tokens)))
    
    # 找到所有plan和reflection标签的位置
    keep_tags = ['plan', 'subtask_list', 'subtask', 'think', 'web_search', 'crawl_page', 'observation', 'subtask_answer']
    keep_tag_indices = []
    
    for i, (segment, _) in enumerate(segment_tokens):
        tag_match = re.match(r'<(/?)([^>]+)>', segment)
        if tag_match:
            is_closing = tag_match.group(1) == '/'
            tag_name = tag_match.group(2)
            if tag_name in keep_tags:
                keep_tag_indices.append((i, tag_name, is_closing))
    
    # 收集需要保留的特殊标签片段索引
    special_indices = set()
    
    # 处理每个特殊标签，确保标签对完整
    for i, tag_name, is_closing in keep_tag_indices:
        if is_closing:
            # 结束标签，查找对应的开始标签
            start_idx = -1
            depth = 0
            for j in range(i, -1, -1):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth += 1
                        else:
                            depth -= 1
                            if depth == 0:
                                start_idx = j
                                break
            if start_idx != -1:
                for idx in range(start_idx, i + 1):
                    special_indices.add(idx)
        else:
            # 开始标签，查找对应的结束标签
            end_idx = -1
            depth = 0
            for j in range(i, len(segments)):
                seg = segment_tokens[j][0]
                m = re.match(r'<(/?)([^>]+)>', seg)
                if m:
                    current_tag = m.group(2)
                    if current_tag == tag_name:
                        if m.group(1) == '/':
                            depth -= 1
                            if depth == 0:
                                end_idx = j
                                break
                        else:
                            depth += 1
            if end_idx != -1:
                for idx in range(i, end_idx + 1):
                    special_indices.add(idx)
    
    # 计算特殊标签部分的token数
    special_segments = [segment_tokens[i] for i in sorted(special_indices)]
    special_tokens = sum(token_count for _, token_count in special_segments)
    
    # 如果特殊标签部分已经超过token限制，截断并返回
    if special_tokens > max_tokens:
        result = []
        current_tokens = 0
        
        for segment, token_count in reversed(special_segments):
            if current_tokens + token_count <= max_tokens:
                result.insert(0, segment)
                current_tokens += token_count
            else:
                if re.match(r'</[^>]+>', segment):
                    tag_name = re.search(r'</([^>]+)>', segment).group(1)
                    start_tag_found = False
                    print(result)
                    for s, _ in reversed(result):
                        if re.match(rf'<{tag_name}>', s):
                            start_tag_found = True
                            break
                    if not start_tag_found:
                        for s, tc in reversed(special_segments):
                            if re.match(rf'<{tag_name}>', s):
                                if current_tokens + tc <= max_tokens:
                                    result.insert(0, s)
                                    current_tokens += tc
                                break
        return ''.join(result) if result else segments[0]
    
    # 还有剩余token，从后往前添加其他内容
    remaining_tokens = max_tokens - special_tokens
    additional_segments = []
    
    # 从后往前遍历所有片段
    for i in range(len(segments) - 1, -1, -1):
        if i in special_indices:
            continue  # 跳过已经保留的特殊标签
        
        segment, token_count = segment_tokens[i]
        
        # 如果添加当前片段会超出限制，尝试部分添加或跳过
        if remaining_tokens <= 0:
            break
        
        if token_count <= remaining_tokens:
            additional_segments.insert(0, segment)
            remaining_tokens -= token_count
        else:
            # 尝试部分添加文本内容
            if not re.match(r'<[^>]+>|</[^>]+>', segment):  # 普通文本
                tokens = tokenizer.encode(segment)
                partial_tokens = tokens[:remaining_tokens]
                partial_text = tokenizer.decode(partial_tokens)
                if partial_text:
                    additional_segments.insert(0, partial_text)
                    remaining_tokens = 0
    
    # 组合结果：特殊标签部分 + 从后往前添加的其他内容
    return ''.join([seg for seg, _ in special_segments] + additional_segments)