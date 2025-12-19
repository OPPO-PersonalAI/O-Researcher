import asyncio
import atexit
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Any
import random

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from dotenv import load_dotenv
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERPER_TIMEOUT = 60
CRAWL_PAGE_TIMEOUT = 500

# --- Cache Backend --- #
class CacheBackend:
    """缓存后端的抽象基类。"""
    def get(self, key: str) -> Optional[str]:
        """根据键获取缓存值。"""
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """设置缓存值，并可选地设置生命周期。"""
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """一个简单的基于内存字典的缓存实现，支持持久化到文件。"""
    def __init__(self, file_path: str):
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self.file_path = file_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Open the file in append mode, creating it if it doesn't exist.
        self._file_handler = open(self.file_path, "a")
        self._load_from_file()
        atexit.register(self._close_file)

    def _load_from_file(self):
        """Loads the cache from a JSONL file."""
        try:
            # We need to read the file first, as it's opened in append mode.
            with open(self.file_path, "r") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        # Each line is a JSON object with one key-value pair
                        entry = json.loads(line)
                        key, value = list(entry.items())[0]
                        self._cache[key] = value
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.warning(
                            f"Skipping malformed line {i+1} in {self.file_path}: {e}"
                        )
            logger.info(
                f"Cache loaded from {self.file_path}, containing {len(self._cache)} items."
            )
        except FileNotFoundError:
            logger.info(
                f"Cache file {self.file_path} not found. A new one will be created."
            )
        except Exception as e:
            logger.warning(f"Could not load cache from {self.file_path}: {e}")

    def _close_file(self):
        """Closes the file handler on exit."""
        if self._file_handler:
            self._file_handler.close()
            logger.info("Cache file handler closed.")

    def get(self, key: str) -> Optional[str]:
        """从缓存中获取值。"""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Sets a value in the cache and appends it to the JSONL file."""
        with self._lock:
            if key not in self._cache:
                self._cache[key] = value
                try:
                    json.dump({key: value}, self._file_handler)
                    self._file_handler.write("\n")
                    self._file_handler.flush()  # Ensure it's written to disk
                except IOError as e:
                    logger.error(f"Could not write to cache file {self.file_path}: {e}")


# --- 代理服务器 --- #
class SerperProxyServer:
    """处理Serper API请求的代理服务类，带有缓存功能。"""
    def __init__(self, cache_backend: CacheBackend):
        self.cache = cache_backend

        server_host = os.environ.get("SERVER_HOST")
        crawl_page_port = os.environ.get("CRAWL_PAGE_PORT")
        self.crawl_page_endpoint = f"http:{server_host}:{crawl_page_port}/crawl_page"
        self.api_key_list = os.environ.get("SERPER_API_KEY","").split('|')
        assert self.api_key_list != [] , "No api keys configured."
        self.serpapi_base_url = os.environ.get("SERPAPI_BASE_URL","")
        assert self.serpapi_base_url != "", "No serpapi_base_url configured."

        # For hit rate logging
        self.total_requests = 0
        self.cache_hits = 0
        self._stats_lock = threading.Lock()
        self.history = []

        # default using first api key
        self._last_api_key_index = 0

    def _generate_cache_key(self, payload: dict) -> str:
        """根据请求负载生成缓存键。"""
        # 对payload进行排序以确保相同内容的不同顺序产生相同的键
        sorted_payload = json.dumps(payload, sort_keys=True)
        return f"serper:{sorted_payload}"

    def _log_hit_rate(self, *, hit: bool):
        """Logs the cache hit rate."""
        with self._stats_lock:
            self.total_requests += 1
            if hit:
                self.cache_hits += 1

            hit_rate = (
                (self.cache_hits / self.total_requests) * 100
                if self.total_requests > 0
                else 0
            )
            status = "HIT" if hit else "MISS"
            logger.info(
                f"Cache {status}. Rate: {hit_rate:.2f}% ({self.cache_hits}/{self.total_requests})"
            )

    def _get_header_case_insensitive(self, headers: dict, header_name: str) -> str:
        """大小写不敏感地获取HTTP头部值"""
        header_name_lower = header_name.lower()
        for key, value in headers.items():
            if key.lower() == header_name_lower:
                return value
        return None

    def _format_results_to_string(self, serper_json: Dict[str, Any], query: str) -> str:
        """Formats the Serper JSON result into a structured string."""
        if "organic" not in serper_json or not serper_json["organic"]:
            return f"No results found for query: '{query}'. Use a less specific query."

        web_snippets = []
        for idx, page in enumerate(serper_json["organic"], 1):
            title = page.get("title", "No Title")
            link = page.get("link", "#")
            date_published = f"\nDate published: {page['date']}" if "date" in page else ""
            source = f"\nSource: {page.get('source', '')}" if "source" in page else ""
            snippet = f"\n{page.get('snippet', '')}".replace("Your browser can't play this video.", "")

            formatted_entry = (
                f"{idx}. [{title}]({link})"
                f"{date_published}{source}"
                f"\n{link}{snippet}"
            )
            web_snippets.append(formatted_entry.strip())
        
        num_results = len(web_snippets)
        return (
            f"A web search for '{query}' found {num_results} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

    def _to_contents_multiqueries(self, search_results: dict):
        """
        将搜索结果字典转换为字符串

        Args:
            search_results: 字典格式的搜索结果，key为查询字符串，value为结果列表
                           value可以是字符串(错误信息)或字典列表(正常搜索结果)

        Returns:
            字符串
        """
        all_contents = []
        total_results = 0

        for query, snippets in search_results.items():
            # 处理报错记录情况（直接返回字符串的情况）
            if isinstance(snippets, str):
                all_contents.append(f"## Query: '{query}'\n{snippets}")
                continue
            # 处理报错情况2 （query太复杂导致搜不到任何东西后，返回空列表）
            elif isinstance(snippets, list):
                if snippets == []:
                    all_contents.append(f"No results found for '{query}'. Try with a more general query, or remove the year filter.")
                    continue

            # 处理正常搜索结果
            web_snippets = []
            idx = 1
            for search_info in snippets:
                if isinstance(search_info, dict):
                    # 确保字典中有必要的键
                    title = search_info.get('title', 'No title')
                    link = search_info.get('link', '#')
                    date = search_info.get('date', '')
                    source = search_info.get('source', '')
                    snippet = search_info.get('snippet', 'No snippet available')

                    redacted_version = (
                        f"{idx}. [{title}]({link})"
                        f"{date}{source}\n"
                        f"{self._pre_visit(link)}{snippet}"
                    ).replace("Your browser can't play this video.", "")

                    web_snippets.append(redacted_version)
                    idx += 1

            query_content = (
                    f"## Query: '{query}'\n"
                    f"Found {len(web_snippets)} results:\n\n"
                    + "\n\n".join(web_snippets)
            )
            all_contents.append(query_content)
            total_results += len(web_snippets)

        # 添加总结果数统计
        if total_results > 0:
            summary = f"# Search Summary\nTotal results: {total_results}\n\n"
            return summary + "\n\n".join(all_contents)
        return "\n\n".join(all_contents)

    def _check_history(self, url_or_query):
        header = ''
        for i in range(len(self.history) - 1, -1, -1):  # Start from the last
            if self.history[i][0] == url_or_query:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                return header
        # 未出现过则加入history
        self.history.append((url_or_query, time.time()))
        return header

    def _pre_visit(self, url):
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == url:
                return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
        return ""
    
    # def _get_api_keys_from_header(self, headers):
    #     """从请求头中获取API Key列表"""
    #     api_key_header_str = self._get_header_case_insensitive(headers, "X-API-KEY")
    #     # logger.info(f"api_key_header_str: {api_key_header_str}")
    #     if not api_key_header_str:
    #         return []
        
    #     api_keys = api_key_header_str.split('|')
    #     if not api_keys:
    #         return []
    #     # logger.debug(f"api_keys: {api_keys}")
        
    #     return api_keys
    
    def _select_api_key_random(self):
        """使用随机策略选择一个API Key"""
        self._last_api_key_index = random.randint(0, len(self.api_key_list) - 1)
        return self.api_key_list[self._last_api_key_index]

    def _select_api_key_with_round_robin(self):
        """使用轮询策略选择API Key"""
        
        # 使用类变量记录上次使用的索引
        if not hasattr(self, '_last_api_key_index'):
            self._last_api_key_index = -1
        
        self._last_api_key_index = (self._last_api_key_index + 1) % len(self.api_key_list)
        return self.api_key_list[self._last_api_key_index]
    
    async def _try_api_request(self, api_request_data, api_key, request_data):
        """尝试使用指定的API Key发送请求"""
        api_headers = {"Content-Type": "application/json", 'X-API-KEY': api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.serpapi_base_url,
                json=api_request_data,
                headers=api_headers,
                timeout=request_data.get("timeout", SERPER_TIMEOUT),
            )
            response.raise_for_status()
            results = response.json()
            
            # 检查结果是否有效
            query = api_request_data.get('q', '')
            filter_year = api_request_data.get('filter_year')
            
            if "organic" not in results:
                return f"No results found for query: '{query}'. Use a less specific query."
            if len(results["organic"]) == 0:
                year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
                return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
                
            return results

    async def process_request(self, request_data: dict, headers: dict) -> dict:
        api_request_data = request_data.copy()
        queries = request_data['q']
        querylist = [query.strip() for query in request_data['q'].split('|') if query.strip()]
        if len(querylist) == 0:
            error_messages = f"查询{queries}分割失败！请严格按照要求：每个查询用'|'分割！"
            return error_messages

        search_results = {}
        total_results = 0
        remaining_queries = len(querylist)
        serp_num=request_data.get('num')
        remaining_serp = serp_num

        seen_set = set()
        unique_results = []
        # if len(querylist) == 0:
        #     raise Exception(f"查询{request_data['q']}分割失败！请严格按照要求：每个查询用'|'分割！")
        # api_request_data['num'] = int(original_num) // len(querylist)
        # results = ""
        for q in querylist:
            try:
                # # 检查历史记录
                # header = self._check_history(q)
                # if "You previously visited this page" in header:
                #     search_results[q] = header
                #     remaining_queries -= 1
                #     continue
                current_serp = max(1, remaining_serp // remaining_queries)
                api_request_data['q'] = q
                api_request_data['num'] = current_serp
                logger.info(f"api_request_data: {api_request_data}")
                snippets = await self.process_request_single(api_request_data, headers)
                
                # 为每个查询创建独立的去重结果
                unique_results_for_query = []
                if isinstance(snippets, str):
                    unique_results_for_query = [snippets]
                else:
                    # 去重处理
                    for result in snippets:
                        if isinstance(result, dict) and 'link' in result:
                            if result['link'] not in seen_set:  # 仍然使用全局seen_set防止跨查询重复
                                seen_set.add(result['link'])
                                unique_results_for_query.append(result)
                    
                # 使用当前查询的独立结果
                search_results[q] = unique_results_for_query[:current_serp]
                total_results += len(search_results[q])
                remaining_serp -= len(search_results[q])
                remaining_queries -= 1
                
            except Exception as e:
                print(f"Error searching for query '{q}': {str(e)}")
                remaining_queries -= 1  # 即使失败也要减少剩余查询计数
        if total_results < serp_num and len(querylist) > 1:
            additional_needed = serp_num - total_results
            # 按查询结果数量升序排序，优先从结果少的查询补充
            sorted_queries = sorted(
                [q for q in querylist if q in search_results and isinstance(search_results[q], list)],
                key=lambda x: len(search_results[x])
            )

            # 修改这部分代码
            for q in sorted_queries:
                if additional_needed <= 0:
                    break
                current_count = len(search_results[q])
                additional = min(additional_needed, 3)
                try:
                    api_request_data['q'] = q
                    api_request_data['num'] = additional
                    extra_snippets = await self.process_request_single(api_request_data, headers)

                    # 添加类型检查 - 确保extra_snippets是列表
                    if isinstance(extra_snippets, str):
                        print(f"警告: 查询 '{q}' 的补充结果为报错: {extra_snippets}，已跳过")
                        continue

                    if not isinstance(extra_snippets, list):
                        print(f"警告: 查询 '{q}' 的补充结果格式不正确，已跳过")
                        continue

                    if extra_snippets:
                        existing_links = {res['link'] for res in search_results[q] if
                                          isinstance(res, dict) and 'link' in res}

                        new_results = []
                        for res in extra_snippets:
                            # 确保结果为字典且包含link字段
                            if isinstance(res, dict) and 'link' in res:
                                link = res['link']
                                if link not in existing_links:
                                    new_results.append(res)
                            else:
                                print(f"警告: 查询 '{q}' 的补充结果中有无效格式条目，已跳过")

                        new_results = new_results[:additional_needed]

                        if new_results:
                            search_results[q].extend(new_results)
                            total_results += len(new_results)
                            additional_needed -= len(new_results)
                except Exception as e:
                    print(f"Error supplementing query '{q}': {str(e)}")
        content = self._to_contents_multiqueries(search_results)

        return content
    async def process_request_single(self, request_data: dict, headers: dict) -> dict:
        """处理Serper API请求，优先从缓存获取，缓存未命中则转发请求。"""
        start_time = time.time()

        # P0: num logic from README
        original_num = request_data.get("num", 10)
        api_request_data = request_data.copy()

        try:
            # Generate cache key based on the API request (10 or 100 results)
            cache_key = self._generate_cache_key(api_request_data)
            # logger.info(f"cache_key: {cache_key}")

            # Check cache
            cached_result_str = self.cache.get(cache_key)
            if cached_result_str:
                self._log_hit_rate(hit=True)
                cached_result = json.loads(cached_result_str)
                # Trim the results to the originally requested number
                if "organic" in cached_result:
                    cached_result["organic"] = cached_result["organic"][:original_num]
                logger.info(
                    f"Request processed from cache in {time.time() - start_time:.2f} seconds."
                )
                results =  cached_result
            else:
                # Cache miss, forward request to Serper API
                self._log_hit_rate(hit=False)
                logger.info(f"Forwarding to Serper API for key: {cache_key}")

                # 从headers中获取API Key列表
                # header_api_keys = self._get_api_keys_from_header(headers)
                # logger.info(f"header_api_keys: {header_api_keys}")

                if not self.api_key_list:
                    # logger.error(f"Invalid headers: {headers}")
                    raise HTTPException(
                        status_code=401, detail="Serper API key not configured"
                    )
                
                # 使用负载均衡选择一个API Key
                api_key = self._select_api_key_random()
                logger.info(f"Choosing api_key: {api_key}")
                try_times = 0

                try:
                    try_times += 1
                    results = await self._try_api_request(api_request_data, api_key, request_data)
                    if isinstance(results, dict):
                        self.cache.set(cache_key, json.dumps(results))
                        logger.info(f"Successfully cached {request_data['q']}")
                except Exception as first_error:
                    logger.warning(f"API Key with ending {api_key[-5:]} failed, trying alternatives")

                    if try_times == len(self.api_key_list):
                        # 没有其他可用的API Key
                        logger.warning(f"API Key with ending {api_key[-5:]} failed, no alternatives left")
                        raise first_error
                    
                    # 尝试剩余的API Keys
                    last_error = first_error
                    while try_times < len(self.api_key_list):
                        try:
                            try_times += 1
                            api_key = self._select_api_key_with_round_robin()
                            logger.info(f"Choosing api_key: {api_key}")
                            results = await self._try_api_request(api_request_data, api_key, request_data)
                            if isinstance(results, dict):
                                self.cache.set(cache_key, json.dumps(results))
                                logger.info(f"Successfully cached {request_data['q']}")
                            # 成功找到可用的API Key
                            logger.info(f"Successfully used alternative API Key with ending {api_key[-5:]}")
                            break
                        except Exception as e:
                            logger.warning(f"API Key with ending {api_key[-5:]} failed, trying alternatives")
                            last_error = e
                    else:
                        # 所有API Keys都失败
                        logger.warning(f"All api key failed.")
                        raise last_error

            web_snippets: List[str] = list()
            idx = 0
            if isinstance(results, dict) and 'organic' in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    _search_result = {
                        "idx": idx,
                        "title": page["title"],
                        "date": date_published,
                        "snippet": snippet,
                        "source": source,
                        "link": page['link']
                    }

                    web_snippets.append(_search_result)
            elif isinstance(results, str):
                web_snippets.append(results)
            return web_snippets

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error calling Serper API: {e.response.status_code} {e.response.text}"
            )

            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"Unexpected error during Serper API request: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"An unexpected error occurred: {str(e)}"
            )


# --- FastAPI 应用设置 --- #

app = FastAPI(title="Serper API Proxy with Cache")

# --- Application Initialization and Shutdown ---
CACHE_FILE = "server/cache/serper_api_cache.jsonl"  # Shared cache with v2
cache_backend = InMemoryCache(file_path=CACHE_FILE)
proxy_server = SerperProxyServer(cache_backend=cache_backend)

# The atexit registration is now handled within the InMemoryCache class


@app.post("/search")
async def serper_proxy_endpoint(request: Request):
    """代理Serper API的搜索端点。"""
    try:
        # logger.info("TEST---------------")
        logger.info(request)
        request_data = await request.json()
        headers = dict(request.headers)

        return await proxy_server.process_request(request_data, headers)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
    except Exception as e:
        # Unhandled exceptions are caught and logged by the proxy_server now,
        # but we keep a general handler here as a fallback.
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Unhandled exception in endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "cache_stats": {
            "total_requests": proxy_server.total_requests,
            "cache_hits": proxy_server.cache_hits,
            "hit_rate": (proxy_server.cache_hits / proxy_server.total_requests * 100) if proxy_server.total_requests > 0 else 0
        }
    }

# --- 主程序入口 --- #

if __name__ == "__main__":
    # 从环境变量或默认值中获取主机和端口
    # 优先从环境变量读取，如果不存在则使用默认值
    host = os.getenv("SERVER_HOST", None)
    port = os.getenv("WEBSEARCH_PORT", None)
    if port == None:
        raise NotImplementedError("[ERROR] WEBSEARCH_PORT NOT SET")
    port = int(port)

    # 启动服务器
    logger.info(f"启动Serper缓存服务器... http://{host}:{port}")
    logger.info(f"Ready to search")
    uvicorn.run(app, host=host, port=port)


'''
source .envrc
python server/cache_serper_server_v3.py

# To test the server:
# python server_tests/test_cache_serper_server_v3.py
'''
