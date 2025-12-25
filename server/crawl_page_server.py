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

# ============ API Configuration Functions ============
def get_summary_api() -> dict:
    """
    Get random Summary API configuration from environment variables.
    Environment variable format:
    - SUMMARY_API_URLS: Multiple URLs separated by |, randomly select one
    - SUMMARY_API_KEYS: Multiple KEYs separated by |, randomly select one
    - SUMMARY_API_MODELS: Multiple model names separated by | (optional, default gpt-5-mini)
    
    Note: URL, KEY, MODEL are independently randomly selected, no need to correspond
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
    
    selected_url = random.choice(urls)
    selected_key = random.choice(keys)
    selected_model = random.choice(models)
    
    return {
        "url": selected_url,
        "key": selected_key,
        "model": selected_model
    }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CRAWL_PAGE_TIMEOUT = 1000

class CrawlPageRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to crawl")
    think_content: str = Field(..., description="Think content for guiding summary or generating click_intent")
    web_search_query: str = Field(..., description="Web search query")

class CrawlPageResponse(BaseModel):
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
        return random.randint(0, len(self.api_key_list) - 1)

    def _select_api_key_with_round_robin(self, api_key_index):
        return (api_key_index + 1) % len(self.api_key_list)
    
    async def _fetch_with_api(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0) -> Tuple[str, str]:
        """
        Fetch single URL using API round-robin.
        Request timeout = timeout seconds.
        Return (content, url) on success; (error_msg, url) on all failures.
        """
        
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
                logger.warning(f"API Key with ending {self.api_key_list[api_key_index][-5:]} failed, no alternatives left")
                raise first_error
            
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
                raise last_error
        return results

    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str, base_delay: float = 1.0, max_delay: float = 16.0, apikey: str = '') -> Tuple[str, str]:
        """
        Fetch single URL with up to self.max_retries attempts.
        Request timeout = timeout seconds.
        Return (content, url) on success; (error_msg, url) on all failures.
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
                        'X-DashScope-DataInspection':'{"input":"disable","output":"disable"}'
                    },
                    stream=False,
                    timeout=self.summary_timeout
                )
                content = completion.choices[0].message.content
                logger.info(f"AI API response received, length: {len(content)} chars")
                return content
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[Attempt {attempt}] AI API call failed: {last_error}, API_KEY: {api_key}, API_URL: {api_url}")
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) # 60s -> 120s -> 240s -> 480s
                    logger.info(f"Waiting {delay:.2f}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retries exhausted, AI API call failed.")

        return f"AI processing failed (after {max_retries} retries): {last_error}"
        
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
            logger.info("--------- Processing crawl_page request ---------")
            logger.info(f"Processing {len(request.urls)} URLs: {request.urls}")
            logger.info(f"web_search_query='{request.web_search_query}'")
            
            urls = self.validate_urls(request.urls)
            if not urls:
                logger.warning("No valid URLs found after validation")
                return CrawlPageResponse(
                    success=False,
                    obs="",
                    error_message="No valid URLs found",
                    processing_time=time.time() - start_time
                )
            
            logger.info(f"Processing {len(urls)} URLs: {urls}")
            
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

            logger.info(f"Page fetching completed after {time.time() - start_time} seconds")
            
            logger.info("Using 'once' strategy - single summarization of all content")
            page_contents = "\n\n".join(f"Page {i+1} [{result[1]}]: {result[0]}" for i, result in enumerate(page_results))
            logger.info(f"Combined content for summarization: {len(page_contents)} chars")
            summary_result = await self.summarize_content(page_contents, request)
            processing_time = time.time() - start_time
            logger.info(f"crawl page done, cost time: {processing_time:.2f} seconds, result length: {len(summary_result)} chars")
            logger.info("--------- Request processed successfully ---------")
            
            return CrawlPageResponse(
                success=True,
                obs=summary_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"crawl page error: {str(e)}", exc_info=True)
            logger.error("--------- Request processing failed ---------")
            return CrawlPageResponse(
                success=False,
                obs="",
                error_message=f"crawl page error: {str(e)}",
                processing_time=processing_time
            )


crawl_server = CrawlPageServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CrawlPage server started")
    yield
    logger.info("CrawlPage server stopped")

app = FastAPI(
    title="CrawlPage Tool Server",
    description="FastAPI-based crawl_page tool service with high concurrency and fault tolerance",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/crawl_page", response_model=CrawlPageResponse)
async def crawl_page_endpoint(request: CrawlPageRequest):
    """
    CrawlPage tool API endpoint
    
    Args:
        - urls: List[str] - Required, list of URLs to crawl
        - think_content: str - Required, think content for guiding summary or generating click_intent
        - web_search_query: str - Required, WebSearch query

    Returns:
        response: CrawlPageResponse with fields:
        - success: bool - Whether successful
        - obs: str - AI summary
        - processing_time: float - Processing time
        - error_message: Optional[str] - Error message (if failed)
    """
    logger.info(f"Received crawl_page request from client")
    try:
        result = await asyncio.wait_for(
            crawl_server.process_crawl_page(request),
            timeout=CRAWL_PAGE_TIMEOUT
        )
        logger.info(f"Request completed successfully, success={result.success}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Request timeout after {CRAWL_PAGE_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"Request timeout: {CRAWL_PAGE_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Endpoint processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/")
async def root():
    return {
        "message": "CrawlPage Tool Server",
        "version": "1.0.0",
        "endpoints": {
            "crawl_page": "/crawl_page",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = os.getenv("CRAWL_PAGE_PORT", "20001")
    is_debug = os.getenv("DEBUG", "false").lower() == "true"

    if port is None:
        raise RuntimeError("[ERROR] CRAWL_PAGE_PORT environment variable not set!")
    
    port = int(port)

    if is_debug:
        print("crawl_page starting in [DEBUG MODE], waiting for debugger to attach...")

        import debugpy
        debugpy.listen(("0.0.0.0", 9527))
        
        debugpy.wait_for_client()
        print("Debugger attached successfully!")

        logger.info(f"[DEBUG MODE] Starting CrawlPage server... http://{host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        logger.info(f"[DEV MODE] Starting CrawlPage server... http://{host}:{port}")
        logger.info("Single process mode, suitable for development")
        logger.info(f"For production use: uvicorn crawl_page_server_v4:app --host {host} --port {port} --workers N")
        
        uvicorn.run(
            "crawl_page_server_v4:app",
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )