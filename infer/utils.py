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

    split_pattern = re.compile(r'(<\/?(?:{})>)'.format('|'.join(ALLOWED_TAGS)))
    segments = split_pattern.split(text)
    segments = [s for s in segments if s.strip()]

    content_buffer = []
    think_content = ''

    for seg in segments:
        if seg.startswith('<'):
            is_close_tag = seg.startswith('</')
            tag_name = seg.strip('<>/').lower()

            if tag_name not in ALLOWED_TAGS:
                content_buffer.append(seg)
                continue

            if not is_close_tag:
                tag_stack.append((tag_name, len(content_buffer)))
            else:
                if not tag_stack:
                    continue
                open_tag, content_start_idx = tag_stack.pop()

                if open_tag == tag_name:
                    paired_content = ''.join(content_buffer[content_start_idx:])
                    tag_pairs.append({
                        "tool": open_tag,
                        "content": paired_content.strip()
                    })
                    content_buffer = content_buffer[:content_start_idx]
                    
                    if "think" in open_tag:
                        think_content = paired_content.strip()
        else:
            content_buffer.append(seg)

    if not tag_pairs:
        return "", "", ""

    last_tag = tag_pairs[-1]
    tool = f"</{last_tag['tool']}>"
    content = last_tag['content']
    
    if last_tag['tool'] == 'web_search':
        query_match = re.search(r'query=(.*?)(&|$)', content)
        if "num=" in content:
            num_match = re.search(r'num=(\d+)', content)
        else:
            num_match = re.search(r'serp_num=(\d+)', content)
        
        queries = query_match.group(1).split('|') if query_match else []
        num = int(num_match.group(1)) if num_match else 10
        
        parsed_content = {
            'queries': queries,
            'num': num,
            'raw': content
        }
        return think_content, tool, parsed_content
        
    elif last_tag['tool'] == 'crawl_page':
        url_match = re.search(r'(.*)', content)
        urls = url_match.group(1).split('|') if url_match else []
        
        parsed_content = {
            'urls': urls,
            'raw': content
        }
        return think_content, tool, parsed_content
        
    elif last_tag['tool'] == 'code_execute':
        code_match = re.search(r'code=(.*)', content, re.DOTALL)
        code = code_match.group(1).strip() if code_match else content

        parsed_content = {
            'code': code,
            'raw': content
        }
        return think_content, tool, parsed_content

    return think_content, tool, content


def extract_last_tag(text, start_tag, end_tag):
    end_index = text.rfind(end_tag)
    if end_index == -1:
        return ""
    start_index = text.rfind(start_tag, 0, end_index)
    if start_index == -1:
        return ""
    start_index += len(start_tag)
    return text[start_index:end_index]

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read JSONL (JSON Lines) file and return data list.
    Each line is parsed as a JSON object and merged into a list.
    
    Args:
        file_path (str): JSONL file path
    
    Returns:
        List[Dict[str, Any]]: List containing all JSON objects
    
    Exceptions:
        Handles file not found and parse errors, returns empty list with warning
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Strip whitespace and parse JSON
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_num} parse failed, skipping entry")
    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
    return data


def write_jsonl(
    data: List[Dict[str, Any]], 
    file_path: str, 
    append: bool = False, 
    ensure_ascii: bool = False
) -> bool:
    """
    Write data list to JSONL file, one JSON object per line.
    Supports append mode and non-ASCII character handling.
    
    Args:
        data (List[Dict[str, Any]]): List of JSON objects to write
        file_path (str): Target file path
        append (bool): Append mode, default False (overwrite)
        ensure_ascii (bool): Escape non-ASCII chars, default False (preserve Unicode)
    
    Returns:
        bool: Whether write succeeded
    
    Example:
        write_jsonl([{"key": "value"}], "data.jsonl")
    """
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        mode = 'a' if append else 'w'
        with open(file_path, mode, encoding='utf-8') as file:
            for item in data:
                # Convert to JSON string ensuring proper encoding
                json_line = json.dumps(item, ensure_ascii=ensure_ascii) + '\n'
                file.write(json_line)
        return True
    except Exception as e:
        print(f"Error writing file: {str(e)}")
        return False


def read_json(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Read standard JSON file and return parsed Python object.
    Supports both object and array format JSON files.
    
    Args:
        file_path (str): JSON file path
    
    Returns:
        Union[Dict[str, Any], List[Dict[str, Any]]]: Parsed Python object
    
    Exceptions:
        Handles file not found, parse errors, returns None with error message
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} does not exist")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None


def write_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    file_path: str, 
    indent: int = 2, 
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> bool:
    """
    Write Python object to standard JSON file.
    Supports dict and list data types.
    
    Args:
        data (Union[Dict[str, Any], List[Dict[str, Any]]]): Python object to write
        file_path (str): Target file path
        indent (int): Indent spaces, default 2 (pretty format)
        ensure_ascii (bool): Escape non-ASCII chars, default False (preserve Unicode)
        sort_keys (bool): Sort keys, default False
    
    Returns:
        bool: Whether write succeeded
    
    Example:
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
        print(f"Error writing file: {str(e)}")
        return False


def count_tokens(text, tokenizer):
    """Count tokens in text using transformers tokenizer"""
    if not text:
        return 0
    return len(tokenizer.encode(text))


def truncate_special_tokens(text: str, max_tokens: int, tokenizer: str) -> str:
    """
    Split text by special tokens, count tokens and keep content from end based on max tokens.
    Only keep <plan>, <|FunctionExecute|> tags and their preceding <think> tag, delete others.
    
    Args:
        text (str): Text containing special tokens
        max_tokens (int): Maximum tokens to keep
        tokenizer (str): Tokenizer for tokenization
    
    Returns:
        str: Truncated text
    """
    
    token_pattern = r'<[^>]+>|</[^>]+>'
    
    segments = re.split(f'({token_pattern})', text)
    segments = [seg for seg in segments if seg]
    
    if not segments:
        return ""
    
    segment_tokens = []
    for segment in segments:
        tokens = tokenizer.encode(segment)
        segment_tokens.append((segment, len(tokens)))
    
    keep_tags = ['plan', 'reflection']
    keep_tag_indices = []
    
    for i, (segment, _) in enumerate(segment_tokens):
        tag_match = re.match(r'<(/?)([^>]+)>', segment)
        if tag_match:
            is_closing = tag_match.group(1) == '/'
            tag_name = tag_match.group(2)
            if tag_name in keep_tags:
                keep_tag_indices.append((i, tag_name, is_closing))
    
    special_indices = set()
    
    for i, tag_name, is_closing in keep_tag_indices:
        if is_closing:
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
    
    special_segments = [segment_tokens[i] for i in sorted(special_indices)]
    special_tokens = sum(token_count for _, token_count in special_segments)

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

    remaining_tokens = max_tokens - special_tokens
    additional_segments = []
    
    for i in range(len(segments) - 1, -1, -1):
        if i in special_indices:
            continue
        
        segment, token_count = segment_tokens[i]

        if remaining_tokens <= 0:
            break
        
        if token_count <= remaining_tokens:
            additional_segments.insert(0, segment)
            remaining_tokens -= token_count
        else:
            if not re.match(r'<[^>]+>|</[^>]+>', segment):
                tokens = tokenizer.encode(segment)
                partial_tokens = tokens[:remaining_tokens]
                partial_text = tokenizer.decode(partial_tokens)
                if partial_text:
                    additional_segments.insert(0, partial_text)
                    remaining_tokens = 0
    
    return ''.join([seg for seg, _ in special_segments] + additional_segments)



def truncate_special_tokens_64k(text: str, max_tokens: int, tokenizer: str) -> str:
    """
    Split text by special tokens, count tokens and keep content from end based on max tokens.
    Only keep <plan>, <|FunctionExecute|> tags and their preceding <think> tag, delete others.
    (64k version with extended tag support)
    
    Args:
        text (str): Text containing special tokens
        max_tokens (int): Maximum tokens to keep
        tokenizer (str): Tokenizer for tokenization
    
    Returns:
        str: Truncated text
    """

    token_pattern = r'<[^>]+>|</[^>]+>'
    
    segments = re.split(f'({token_pattern})', text)
    segments = [seg for seg in segments if seg]
    
    if not segments:
        return ""
    
    segment_tokens = []
    for segment in segments:
        tokens = tokenizer.encode(segment)
        segment_tokens.append((segment, len(tokens)))
    
    keep_tags = ['plan', 'subtask_list', 'subtask', 'think', 'web_search', 'crawl_page', 'observation', 'subtask_answer']
    keep_tag_indices = []
    
    for i, (segment, _) in enumerate(segment_tokens):
        tag_match = re.match(r'<(/?)([^>]+)>', segment)
        if tag_match:
            is_closing = tag_match.group(1) == '/'
            tag_name = tag_match.group(2)
            if tag_name in keep_tags:
                keep_tag_indices.append((i, tag_name, is_closing))
    
    special_indices = set()
    
    for i, tag_name, is_closing in keep_tag_indices:
        if is_closing:
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
    
    special_segments = [segment_tokens[i] for i in sorted(special_indices)]
    special_tokens = sum(token_count for _, token_count in special_segments)
    
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
    
    remaining_tokens = max_tokens - special_tokens
    additional_segments = []
    
    for i in range(len(segments) - 1, -1, -1):
        if i in special_indices:
            continue
        
        segment, token_count = segment_tokens[i]
        
        if remaining_tokens <= 0:
            break
        
        if token_count <= remaining_tokens:
            additional_segments.insert(0, segment)
            remaining_tokens -= token_count
        else:
            if not re.match(r'<[^>]+>|</[^>]+>', segment):
                tokens = tokenizer.encode(segment)
                partial_tokens = tokens[:remaining_tokens]
                partial_text = tokenizer.decode(partial_tokens)
                if partial_text:
                    additional_segments.insert(0, partial_text)
                    remaining_tokens = 0
    
    return ''.join([seg for seg, _ in special_segments] + additional_segments)