import asyncio
import concurrent.futures
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union
import random

import debugpy
import aiohttp
import requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from dotenv import load_dotenv
load_dotenv(override=True)

import json
from dataclasses import dataclass

# ============ API配置函数 ============
def get_summary_api() -> dict:
    """
    从环境变量获取随机的 Summary API 配置
    环境变量格式：
    - SUMMARY_API_URLS: 多个URL用|分隔，随机选择一个
    - SUMMARY_API_KEYS: 多个KEY用|分隔，随机选择一个
    - SUMMARY_API_MODELS: 多个模型名用|分隔，随机选择一个（可选，默认gpt-5-mini）
    
    注意：URL、KEY、MODEL 会独立随机选择，不需要一一对应
    """
    import random
    
    summary_urls_str = os.environ.get("SUMMARY_API_URLS", "")
    summary_keys_str = os.environ.get("SUMMARY_API_KEYS", "")
    summary_models_str = os.environ.get("SUMMARY_API_MODELS", "gpt-5-mini")
    
    if not summary_urls_str or not summary_keys_str:
        raise ValueError("SUMMARY_API_URLS and SUMMARY_API_KEYS environment variables must be set")
    
    urls = [url.strip() for url in summary_urls_str.split("|") if url.strip()]
    keys = [key.strip() for key in summary_keys_str.split("|") if key.strip()]
    models = [model.strip() for model in summary_models_str.split("|") if model.strip()]
    
    if not urls or not keys:
        raise ValueError("SUMMARY_API_URLS and SUMMARY_API_KEYS cannot be empty")
    
    if not models:
        models = ["gpt-5-mini"]
    
    # 从各自的池中随机选择，实现灵活的负载均衡
    selected_url = random.choice(urls)
    selected_key = random.choice(keys)
    selected_model = random.choice(models)
    
    return {
        "url": selected_url,
        "key": selected_key,
        "model": selected_model
    }

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CRAWL_PAGE_TIMEOUT = 1000

class CrawlPageRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to crawl")
    think_content: str = Field(..., description="思考内容，用于指导总结, 或者基于 think_content 生成 click_intent")
    web_search_query: str = Field(..., description="Web search query")

class CrawlPageResponse(BaseModel):
    """响应模型"""
    success: bool
    obs: str
    error_message: Optional[str] = None
    processing_time: float

class CrawlPageServer:
    def __init__(self):
        logger.info("Initializing CrawlPageServer")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.jina_timeout = 30
        self.summary_timeout = 600
        self.max_retries = 5
        self.jina_token_budget = 80000
        self.api_key_list = os.environ.get("JINA_API_KEY","").split("|")
        logger.info(f"API_KEY_LIST: {self.api_key_list}")
        assert self.api_key_list != [], "No api key configured."
        logger.info(f"CrawlPageServer initialized with jina_timeout={self.jina_timeout}s, summary_timeout={self.summary_timeout}s, token_budget={self.jina_token_budget}")
    
    def _select_api_key_random(self):
        """使用随机策略选择一个API Key"""
        return random.randint(0, len(self.api_key_list) - 1)

    def _select_api_key_with_round_robin(self, api_key_index):
        """使用轮询策略选择API Key"""
        return (api_key_index + 1) % len(self.api_key_list)
    
    async def _fetch_with_api(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0) -> Tuple[str, str]:
        """
        对单个 URL 做使用api轮询抓取。
        每次请求超时 = timeout 秒。
        成功立即返回 (content, url)；全部失败返回 (error_msg, url)。
        """
        
        # 使用负载均衡选择一个API Key
        api_key_index = self._select_api_key_random()
        logger.info(f"Choosing api_key: {self.api_key_list[api_key_index]}")
        try_times = 0
        try:
            try_times += 1
            results = await self._fetch_with_retry(session, url, base_delay, max_delay, self.api_key_list[api_key_index])
            if isinstance(results,tuple) and "[Page content not accessible" in results[0]:
                raise Exception("Unsuccessful crawl")
            logger.info(f"Successfully crawled page with api_key ending with {self.api_key_list[api_key_index][-5:]}")
        except Exception as first_error:
            logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, trying alternatives")

            if try_times == len(self.api_key_list):
                # 没有其他可用的API Key
                logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, no alternatives left")
                raise first_error
            
            # 尝试剩余的API Keys
            last_error = first_error
            while try_times < len(self.api_key_list):
                try:
                    try_times += 1
                    api_key_index = self._select_api_key_with_round_robin(api_key_index)
                    logger.info(f"Choosing api_key: {self.api_key_list[api_key_index]}")
                    results = await self._fetch_with_retry(session, url, base_delay, max_delay, self.api_key_list[api_key_index])
                    if isinstance(results,tuple) and "[Page content not accessible" in results[0]:
                        raise Exception("Unsuccessful crawl")
                    logger.info(f"Successfully crawled page with api_key ending with {self.api_key_list[api_key_index][-5:]}")
                    break
                except Exception as e:
                    logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, trying alternatives")
                    last_error = e
            else:
                # 所有API Keys都失败
                raise last_error
        return results

    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0, apikey: str = '') -> Tuple[str, str]:
        """
        对单个 URL 做最多 self.max_retries 次抓取。
        每次请求超时 = timeout 秒。
        成功立即返回 (content, url)；全部失败返回 (error_msg, url)。
        """
        assert apikey != "", "No api key when fetching."
        attempt = 0
        last_exc = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(f"[Attempt {attempt}/{self.max_retries}] {url}")
                timeout = aiohttp.ClientTimeout(total=self.jina_timeout)
                jina_url = f"https://r.jina.ai/{url}"
                headers = {
                    'Authorization': f'Bearer {apikey}',
                    'X-Engine': 'browser',
                    'X-Return-Format': 'text',
                    "X-Remove-Selector": "header, .class, #id",
                    'X-Timeout': str(self.jina_timeout),
                    "X-Retain-Images": "none",
                    'X-Token-Budget': "80000"
                }

                async with session.get(jina_url, headers=headers, timeout=timeout) as resp:
                    resp.raise_for_status()
                    content = await resp.text()
                    return content, url
            except asyncio.TimeoutError:
                last_exc = f"Timeout after {self.jina_timeout}s"
                logger.warning(f"[Attempt {attempt}] Timeout for {url}")
            except Exception as e:
                last_exc = str(e)
                logger.warning(f"[Attempt {attempt}] Error for {url}: {e}")

            if attempt < self.max_retries:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                await asyncio.sleep(delay)

        return f"[Page content not accessible: {last_exc}]", url

    async def read_page_async(self, session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
        return await self._fetch_with_api(session, url)


    def validate_urls(self, urls: List[str]) -> List[str]:
        """Validate HTTP/HTTPS URLs."""
        processed_urls = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            if url.startswith(('http://', 'https://')):
                processed_urls.append(url)
            else:
                logger.warning(f"Invalid URL format (must start with http:// or https://): {url}")        
        return processed_urls

    def get_summary_prompt(self, web_search_query: str, think_content: str, page_contents: str) -> str:
        """构建网页总结提示"""
        prompt = f"""
        Target: Extract all content from a web page that matches a specific web search query and search query, ensuring completeness and relevance. (No response/analysis required.)
        
        web search query: 
        {web_search_query}

        Clues and ideas: 
        {think_content}
        
        Searched Web Page: 
        {page_contents}

        Important Notes:
        - Summarize all content (text, tables, lists, code blocks) into concise points that directly address query and clues and ideas.
        - Preserve and list all relevant links ([text](url)) from the web page.
        - Summarize in three points: web search query-related information, clues and ideas-related information, and relevant links with descriptions.
        - If no relevant information exists, Just output "No relevant information"
        - Strictly follow the Markdown format below to organize and present your findings. Do not add any introductory or concluding remarks.
            # 1.web search query related information
            [Here, list the keywords, long-tail phrases, or questions you would use in a search engine to find relevant information.]
            # 2.clues and ideas related information
            [Here, list the core concepts, different analytical perspectives, potential sub-topics to explore, or creative ideas related to the theme.]
            # 3.relevant links with descriptions 
            [Here, provide more than one high-quality, relevant web links, each with a brief descriptions of its core content.]
        """
        return prompt

    async def call_ai_api_async(self, system_prompt: str, user_prompt: str, max_retries: int = 5, base_delay: float = 60) -> str:
        """异步调用AI API，支持重试机制"""
        selected_summary_api = get_summary_api()
        api_url, api_key, model = selected_summary_api["url"], selected_summary_api["key"], selected_summary_api["model"]
        logger.info(f"Calling AI API with model: {model}, API URL: {api_url}, max_retries: {max_retries}")
        attempt = 0
        last_error = None

        while attempt < max_retries:
            attempt += 1
            try:
                logger.info(f"[Attempt {attempt}/{max_retries}] Calling AI API...")
                client = AsyncOpenAI(base_url=api_url, api_key=api_key)
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    extra_headers={
                        'X-DashScope-DataInspection':'{"input":"disable","output":"disable"}'  # 关键参数(接口输入和输出的信息是否通过滤网过滤)
                    },
                    stream=False,
                    timeout=self.summary_timeout
                )
                content = completion.choices[0].message.content
                logger.info(f"AI API response received, length: {len(content)} chars")
                return content
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[Attempt {attempt}] AI API调用失败: {last_error}, API_KEY: {api_key}, API_URL: {api_url}")
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) # 60s -> 120s -> 240s -> 480s
                    logger.info(f"等待 {delay:.2f}s 后重试...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("所有重试次数已用完，AI API调用失败。")

        return f"AI处理失败（重试{max_retries}次后）: {last_error}"
        
    async def summarize_content(self, content: str, request: CrawlPageRequest) -> str:
        """Helper function to summarize content"""
        logger.info(f"Summarizing content of length: {len(content)} chars")
        content = content[:60000]
        detailed_prompt = self.get_summary_prompt(
            request.web_search_query, request.think_content, content
        )
        return await self.call_ai_api_async(
            "You are a summary agent robot.", detailed_prompt
        )

    async def process_crawl_page(self, request: CrawlPageRequest) -> CrawlPageResponse:
        start_time = time.time()
        try:
            # 直接使用传入的参数
            logger.info("--------- 开始处理crawl_page请求 ---------")
            logger.info(f"Processing {len(request.urls)} URLs: {request.urls}")
            logger.info(f"web_search_query='{request.web_search_query}'")
            
            # 验证和清理URL列表
            urls = self.validate_urls(request.urls)
            if not urls:
                logger.warning("No valid URLs found after validation")
                return CrawlPageResponse(
                    success=False,
                    obs="",
                    error_message="没有找到有效的URL",
                    processing_time=time.time() - start_time
                )
            
            logger.info(f"开始处理{len(urls)}个URL: {urls}")
            
            # 异步获取页面内容
            page_contents = ""
            logger.info("Creating aiohttp session for page fetching")
            async with aiohttp.ClientSession() as session:
                tasks = [self.read_page_async(session, url) for url in urls]
                logger.info(f"Fetching {len(tasks)} pages concurrently")
                page_results = await asyncio.gather(*tasks, return_exceptions=True)

                processed_results = []
                for i, result in enumerate(page_results):
                    if isinstance(result, Exception):
                        logger.error(f"Unhandled exception for URL {urls[i]}: {result}")
                        processed_results.append((f"[Page content not accessible: {result}]", urls[i]))
                    else:
                        processed_results.append(result)
                page_results = processed_results

            ##### 结束 Jina read page #####
            logger.info(f"Page fetching completed after {time.time() - start_time} seconds")
            
            ##### 开始 page summary #####
            # Single summarization of all content
            logger.info("Using 'once' strategy - single summarization of all content")
            page_contents = "\n\n".join(f"Page {i+1} [{result[1]}]: {result[0]}" for i, result in enumerate(page_results))
            logger.info(f"Combined content for summarization: {len(page_contents)} chars")
            summary_result = await self.summarize_content(page_contents, request)
            processing_time = time.time() - start_time
            logger.info(f"crawl page done, cost time: {processing_time:.2f} seconds, result length: {len(summary_result)} chars")
            logger.info("--------- 请求处理成功 ---------")
            
            return CrawlPageResponse(
                success=True,
                obs=summary_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"crawl page error: {str(e)}", exc_info=True)
            logger.error("--------- 请求处理失败 ---------")
            return CrawlPageResponse(
                success=False,
                obs="",
                error_message=f"crawl page error: {str(e)}",
                processing_time=processing_time
            )


# 创建服务器实例
crawl_server = CrawlPageServer()

# 创建FastAPI应用
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的初始化
    logger.info("CrawlPage服务器启动")
    yield
    # 关闭时的清理
    logger.info("CrawlPage服务器关闭")

app = FastAPI(
    title="CrawlPage工具服务器",
    description="基于FastAPI的crawl_page工具服务，支持高并发和容错",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/crawl_page", response_model=CrawlPageResponse)
async def crawl_page_endpoint(request: CrawlPageRequest):
    logger.info(f"Received crawl_page request from client")
    """
    CrawlPage工具接口
    
    参数：
    # 参数
    - urls: List[str] - 必须, 要爬取的URL列表
    - think_content: str - 必须, 思考内容，用于指导总结, 或者基于 think_content 生成 click_intent
    - web_search_query: str - 必须, WebSearch query

    # API KEY
    - api_url: str - 必须, AI API URL
    - api_key: str - 必须, AI API Key
    - model: str - 必须, AI 模型
    

    返回值:
    response: CrawlPageResponse, 包含四个字段
    - success: bool - 是否成功
    - obs: str - AI总结
    - processing_time: float - 处理时间
    - error_message: Optional[str] - 错误信息（如果失败）
    """
    try:
        result = await asyncio.wait_for(
            crawl_server.process_crawl_page(request),
            timeout=CRAWL_PAGE_TIMEOUT
        )
        logger.info(f"Request completed successfully, success={result.success}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {CRAWL_PAGE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"请求超时: {CRAWL_PAGE_TIMEOUT}秒")
    except Exception as e:
        logger.error(f"接口处理异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    logger.info("Health check requested")
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/")
async def root():
    return {
        "message": "CrawlPage工具服务器",
        "version": "1.0.0",
        "endpoints": {
            "crawl_page": "/crawl_page",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    # 从环境变量获取配置
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = os.getenv("CRAWL_PAGE_PORT", "20001")
    is_debug = os.getenv("DEBUG", "false").lower() == "true"

    if port is None:
        raise RuntimeError("[ERROR] CRAWL_PAGE_PORT 环境变量未设置!")
    
    port = int(port)

    # 判断是否为debug模式
    if is_debug:
        # --- 调试模式配置 ---
        print("crawl_page以【调试模式】启动，等待调试器附加...")

        import debugpy
        debugpy.listen(("0.0.0.0", 9527))
        
        # 暂停脚本，直到调试客户端(VS code)连接上来
        debugpy.wait_for_client()
        print("调试器已成功附加！")

        # 调试模式：单进程，支持热重载
        logger.info(f"[调试模式] 启动CrawlPage服务器... http://{host}:{port}")
        uvicorn.run(
            app,  # 直接传递 app 对象
            host=host,
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        # 生产模式：提示使用命令行启动
        logger.info(f"[开发模式] 启动CrawlPage服务器... http://{host}:{port}")
        logger.info("单进程模式，适用于开发调试")
        logger.info(f"生产环境请使用: uvicorn crawl_page_server_v4:app --host {host} --port {port} --workers N")
        
        # 使用字符串引用 app，支持热重载
        uvicorn.run(
            "crawl_page_server_v4:app",  # 修改：使用字符串引用（假设文件名为 crawl_page_server_v4.py）
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )