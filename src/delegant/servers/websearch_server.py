"""
Delegant WebSearch Server Implementation
=======================================

MCP server implementation for web search operations with context-aware functionality.
Supports multiple search providers with rate limiting and result caching.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus
from pydantic import Field, field_validator
import httpx
import logging

from ..server import HTTPMCPServer, MCPTool
from ..exceptions import ToolExecutionError, ConfigurationError
from ..config import get_config

logger = logging.getLogger(__name__)


class SearchResult(dict):
    """Structured search result with standardized fields."""
    
    def __init__(self, **data):
        # Standardize fields across different search providers
        standardized = {
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "description": data.get("description", data.get("snippet", "")),
            "published_date": data.get("published_date"),
            "source": data.get("source", ""),
            "score": data.get("score", 0.0),
            "image_url": data.get("image_url"),
            "metadata": data.get("metadata", {})
        }
        super().__init__(**standardized)


class WebSearchServer(HTTPMCPServer):
    """Server for web search operations with context awareness.
    
    Supports multiple search providers including DuckDuckGo, Google Custom Search,
    and Bing Search API. Provides intelligent result filtering and caching.
    
    Example:
        server = WebSearchServer(
            provider="duckduckgo",
            max_results=20,
            safe_search=True
        )
        
        results = await server.search("Python programming tutorials")
        images = await server.search_images("data visualization charts")
    """
    
    provider: str = Field(
        default="duckduckgo",
        description="Search provider (duckduckgo, google, bing)"
    )
    api_key: Optional[str] = Field(
        None, 
        description="API key for the search provider (if required)"
    )
    custom_search_engine_id: Optional[str] = Field(
        None,
        description="Custom Search Engine ID for Google (if using Google)"
    )
    max_results: int = Field(
        default=10, 
        description="Maximum search results per query",
        ge=1,
        le=100
    )
    timeout: int = Field(
        default=60, 
        description="Request timeout in seconds",
        ge=5,
        le=300
    )
    safe_search: bool = Field(
        default=True,
        description="Enable safe search filtering"
    )
    language: str = Field(
        default="en",
        description="Search language code (en, es, fr, de, etc.)"
    )
    region: str = Field(
        default="us",
        description="Search region/country code (us, uk, ca, etc.)"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching to reduce API calls"
    )
    cache_duration: int = Field(
        default=3600,
        description="Cache duration in seconds",
        ge=60
    )
    
    # Private attributes
    _cache: Dict[str, Dict[str, Any]] = {}
    _rate_limiter: Dict[str, datetime] = {}
    _rate_limit_delay: float = 1.0  # Minimum delay between requests
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate search provider is supported."""
        supported_providers = ['duckduckgo', 'google', 'bing']
        if v.lower() not in supported_providers:
            raise ValueError(f"Provider must be one of: {supported_providers}")
        return v.lower()
    
    def __init__(self, **data):
        # Set base_url based on provider
        provider = data.get('provider', 'duckduckgo').lower()
        if 'base_url' not in data:
            data['base_url'] = self._get_provider_base_url(provider)
        
        super().__init__(**data)
        
        # Validate provider-specific requirements
        self._validate_provider_config()
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _get_provider_base_url(self, provider: str) -> str:
        """Get base URL for the search provider."""
        provider_urls = {
            'duckduckgo': 'https://api.duckduckgo.com',
            'google': 'https://www.googleapis.com/customsearch/v1',
            'bing': 'https://api.bing.microsoft.com/v7.0/search'
        }
        return provider_urls.get(provider, 'https://api.duckduckgo.com')
    
    def _validate_provider_config(self) -> None:
        """Validate provider-specific configuration."""
        if self.provider == 'google':
            if not self.api_key or not self.custom_search_engine_id:
                raise ConfigurationError(
                    config_source="google_search_config",
                    suggested_fix="Google search requires both api_key and custom_search_engine_id"
                )
        elif self.provider == 'bing':
            if not self.api_key:
                raise ConfigurationError(
                    config_source="bing_search_config",
                    suggested_fix="Bing search requires an api_key"
                )
    
    def _register_builtin_tools(self) -> None:
        """Register built-in search tools."""
        
        # Web search tool
        self.register_tool(MCPTool(
            name="search",
            description="Search the web for information",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": self.max_results,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "language": {
                        "type": "string",
                        "description": "Search language code",
                        "default": self.language
                    },
                    "region": {
                        "type": "string",
                        "description": "Search region code",
                        "default": self.region
                    }
                },
                "required": ["query"]
            },
            execution_func=self.search
        ))
        
        # Image search tool
        self.register_tool(MCPTool(
            name="search_images",
            description="Search for images on the web",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Image search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of image results",
                        "default": min(self.max_results, 20),
                        "minimum": 1,
                        "maximum": 50
                    },
                    "image_size": {
                        "type": "string",
                        "description": "Image size filter (small, medium, large, any)",
                        "default": "any"
                    },
                    "image_type": {
                        "type": "string",
                        "description": "Image type filter (photo, clipart, line, any)",
                        "default": "any"
                    }
                },
                "required": ["query"]
            },
            execution_func=self.search_images
        ))
        
        # News search tool
        self.register_tool(MCPTool(
            name="search_news",
            description="Search for recent news articles",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "News search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of news results",
                        "default": self.max_results,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range for news (day, week, month, year)",
                        "default": "week"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort results by (relevance, date)",
                        "default": "relevance"
                    }
                },
                "required": ["query"]
            },
            execution_func=self.search_news
        ))
    
    async def _rate_limit_check(self) -> None:
        """Check and enforce rate limiting."""
        now = datetime.now()
        provider_key = f"{self.provider}_last_request"
        
        if provider_key in self._rate_limiter:
            time_since_last = (now - self._rate_limiter[provider_key]).total_seconds()
            if time_since_last < self._rate_limit_delay:
                sleep_time = self._rate_limit_delay - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self._rate_limiter[provider_key] = now
    
    def _get_cache_key(self, operation: str, **params) -> str:
        """Generate cache key for the operation and parameters."""
        # Create a stable hash of the operation and parameters
        cache_data = {"operation": operation, "provider": self.provider, **params}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        if not self.enable_caching or cache_key not in self._cache:
            return None
        
        cache_entry = self._cache[cache_key]
        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        
        if (datetime.now() - cache_time).total_seconds() > self.cache_duration:
            # Cache expired
            del self._cache[cache_key]
            return None
        
        logger.debug(f"Cache hit for key: {cache_key}")
        return cache_entry["data"]
    
    def _set_cached_result(self, cache_key: str, data: Any) -> None:
        """Store result in cache."""
        if not self.enable_caching:
            return
        
        self._cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Clean old cache entries if cache is getting large
        if len(self._cache) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self._cache.items():
            cache_time = datetime.fromisoformat(entry["timestamp"])
            if (now - cache_time).total_seconds() > self.cache_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    async def search(
        self, 
        query: str,
        max_results: Optional[int] = None,
        language: Optional[str] = None,
        region: Optional[str] = None
    ) -> List[SearchResult]:
        """Search the web with context about search purpose.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            language: Search language code
            region: Search region code
            
        Returns:
            List of search results with structured data
        """
        max_results = max_results or self.max_results
        language = language or self.language
        region = region or self.region
        
        # Check cache first
        cache_key = self._get_cache_key("search", query=query, max_results=max_results, 
                                       language=language, region=region)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Rate limiting
            await self._rate_limit_check()
            
            # Perform search based on provider
            if self.provider == "duckduckgo":
                results = await self._search_duckduckgo(query, max_results, language, region)
            elif self.provider == "google":
                results = await self._search_google(query, max_results, language, region)
            elif self.provider == "bing":
                results = await self._search_bing(query, max_results, language, region)
            else:
                raise ToolExecutionError(
                    tool_name="search",
                    server_name=self._connection.server_id if self._connection else "websearch",
                    parameters={"query": query},
                    original_error=ValueError(f"Unsupported provider: {self.provider}")
                )
            
            # Cache results
            self._set_cached_result(cache_key, results)
            
            logger.info(f"Web search completed: '{query}' ({len(results)} results)")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            raise
    
    async def _search_duckduckgo(
        self, 
        query: str, 
        max_results: int,
        language: str,
        region: str
    ) -> List[SearchResult]:
        """Perform search using DuckDuckGo API."""
        try:
            # DuckDuckGo Instant Answer API
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
                "no_redirect": "1",
                "safe_search": "1" if self.safe_search else "0"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
            
            results = []
            
            # Process abstract (main result)
            if data.get("Abstract"):
                results.append(SearchResult(
                    title=data.get("Heading", query),
                    url=data.get("AbstractURL", ""),
                    description=data.get("Abstract", ""),
                    source=data.get("AbstractSource", ""),
                    score=1.0
                ))
            
            # Process related topics
            for topic in data.get("RelatedTopics", [])[:max_results-len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(SearchResult(
                        title=topic.get("Text", "").split(" - ")[0],
                        url=topic.get("FirstURL", ""),
                        description=topic.get("Text", ""),
                        source="DuckDuckGo",
                        score=0.8
                    ))
            
            # If we don't have enough results, use HTML scraping fallback
            if len(results) < max_results:
                fallback_results = await self._search_duckduckgo_html(query, max_results - len(results))
                results.extend(fallback_results)
            
            return results[:max_results]
            
        except Exception as e:
            raise ToolExecutionError(
                tool_name="search",
                server_name=self._connection.server_id if self._connection else "websearch",
                parameters={"query": query, "provider": "duckduckgo"},
                original_error=e
            )
    
    async def _search_duckduckgo_html(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback HTML scraping for DuckDuckGo (simplified)."""
        # Note: This is a simplified implementation
        # In production, you might want to use a proper HTML parser
        results = []
        
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(search_url)
                response.raise_for_status()
                
            # This is a placeholder - in practice you'd parse the HTML
            # For now, return empty results to avoid parsing complexity
            return results
            
        except Exception:
            return results
    
    async def _search_google(
        self, 
        query: str, 
        max_results: int,
        language: str,
        region: str
    ) -> List[SearchResult]:
        """Perform search using Google Custom Search API."""
        if not self.api_key or not self.custom_search_engine_id:
            raise ConfigurationError(
                config_source="google_search",
                suggested_fix="Google search requires api_key and custom_search_engine_id"
            )
        
        try:
            params = {
                "key": self.api_key,
                "cx": self.custom_search_engine_id,
                "q": query,
                "num": min(max_results, 10),  # Google API max is 10 per request
                "hl": language,
                "gl": region,
                "safe": "active" if self.safe_search else "off"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    description=item.get("snippet", ""),
                    source=item.get("displayLink", ""),
                    published_date=item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time"),
                    image_url=item.get("pagemap", {}).get("cse_image", [{}])[0].get("src"),
                    score=1.0,
                    metadata={
                        "html_title": item.get("htmlTitle", ""),
                        "html_snippet": item.get("htmlSnippet", ""),
                        "file_format": item.get("fileFormat"),
                        "mime": item.get("mime")
                    }
                ))
            
            return results
            
        except Exception as e:
            raise ToolExecutionError(
                tool_name="search",
                server_name=self._connection.server_id if self._connection else "websearch",
                parameters={"query": query, "provider": "google"},
                original_error=e
            )
    
    async def _search_bing(
        self, 
        query: str, 
        max_results: int,
        language: str,
        region: str
    ) -> List[SearchResult]:
        """Perform search using Bing Search API."""
        if not self.api_key:
            raise ConfigurationError(
                config_source="bing_search",
                suggested_fix="Bing search requires an api_key"
            )
        
        try:
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {
                "q": query,
                "count": min(max_results, 50),  # Bing API max is 50
                "mkt": f"{language}-{region}",
                "safeSearch": "Strict" if self.safe_search else "Off"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
            
            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    description=item.get("snippet", ""),
                    source=item.get("displayUrl", ""),
                    published_date=item.get("dateLastCrawled"),
                    score=1.0,
                    metadata={
                        "about": item.get("about", []),
                        "deep_links": item.get("deepLinks", [])
                    }
                ))
            
            return results
            
        except Exception as e:
            raise ToolExecutionError(
                tool_name="search",
                server_name=self._connection.server_id if self._connection else "websearch",
                parameters={"query": query, "provider": "bing"},
                original_error=e
            )
    
    async def search_images(
        self,
        query: str,
        max_results: Optional[int] = None,
        image_size: str = "any",
        image_type: str = "any"
    ) -> List[SearchResult]:
        """Search for images with contextual filters.
        
        Args:
            query: Image search query
            max_results: Maximum number of results
            image_size: Size filter (small, medium, large, any)
            image_type: Type filter (photo, clipart, line, any)
            
        Returns:
            List of image search results
        """
        max_results = max_results or min(self.max_results, 20)
        
        # Check cache
        cache_key = self._get_cache_key("search_images", query=query, max_results=max_results,
                                       image_size=image_size, image_type=image_type)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            await self._rate_limit_check()
            
            results = []
            
            if self.provider == "google":
                results = await self._search_google_images(query, max_results, image_size, image_type)
            elif self.provider == "bing":
                results = await self._search_bing_images(query, max_results, image_size, image_type)
            else:
                # DuckDuckGo and others don't have image API, return empty
                logger.warning(f"Image search not supported for provider: {self.provider}")
                results = []
            
            self._set_cached_result(cache_key, results)
            
            logger.info(f"Image search completed: '{query}' ({len(results)} results)")
            return results
            
        except Exception as e:
            logger.error(f"Image search failed for query '{query}': {e}")
            raise
    
    async def _search_google_images(
        self, 
        query: str, 
        max_results: int,
        image_size: str,
        image_type: str
    ) -> List[SearchResult]:
        """Search Google Images using Custom Search API."""
        if not self.api_key or not self.custom_search_engine_id:
            return []
        
        size_map = {"small": "icon", "medium": "medium", "large": "xxlarge", "any": ""}
        type_map = {"photo": "photo", "clipart": "clipart", "line": "lineart", "any": ""}
        
        params = {
            "key": self.api_key,
            "cx": self.custom_search_engine_id,
            "q": query,
            "searchType": "image",
            "num": min(max_results, 10),
            "safe": "active" if self.safe_search else "off"
        }
        
        if size_map.get(image_size):
            params["imgSize"] = size_map[image_size]
        if type_map.get(image_type):
            params["imgType"] = type_map[image_type]
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("items", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                description=item.get("snippet", ""),
                image_url=item.get("link", ""),
                source=item.get("displayLink", ""),
                metadata={
                    "thumbnail": item.get("image", {}).get("thumbnailLink"),
                    "width": item.get("image", {}).get("width"),
                    "height": item.get("image", {}).get("height"),
                    "context_link": item.get("image", {}).get("contextLink")
                }
            ))
        
        return results
    
    async def _search_bing_images(
        self, 
        query: str, 
        max_results: int,
        image_size: str,
        image_type: str
    ) -> List[SearchResult]:
        """Search Bing Images using Search API."""
        if not self.api_key:
            return []
        
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": min(max_results, 35),  # Bing Images API max
            "safeSearch": "Strict" if self.safe_search else "Off"
        }
        
        if image_size != "any":
            params["size"] = image_size
        if image_type != "any":
            params["imageType"] = image_type
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://api.bing.microsoft.com/v7.0/images/search",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("value", []):
            results.append(SearchResult(
                title=item.get("name", ""),
                url=item.get("webSearchUrl", ""),
                description=item.get("name", ""),
                image_url=item.get("contentUrl", ""),
                source=item.get("hostPageDisplayUrl", ""),
                metadata={
                    "thumbnail": item.get("thumbnailUrl"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "encoding_format": item.get("encodingFormat"),
                    "content_size": item.get("contentSize")
                }
            ))
        
        return results
    
    async def search_news(
        self,
        query: str,
        max_results: Optional[int] = None,
        time_range: str = "week",
        sort_by: str = "relevance"
    ) -> List[SearchResult]:
        """Search for recent news articles.
        
        Args:
            query: News search query
            max_results: Maximum number of results
            time_range: Time range (day, week, month, year)
            sort_by: Sort by relevance or date
            
        Returns:
            List of news article results
        """
        max_results = max_results or self.max_results
        
        # For this implementation, we'll use regular search with news-specific parameters
        # In production, you might want to use dedicated news APIs
        news_query = f"{query} news"
        
        results = await self.search(news_query, max_results)
        
        # Filter and enhance results for news content
        news_results = []
        for result in results:
            # Simple heuristic to identify news content
            if any(term in result["url"].lower() or term in result["source"].lower() 
                   for term in ["news", "reuters", "cnn", "bbc", "ap", "bloomberg"]):
                news_results.append(result)
        
        return news_results[:max_results]
    
    async def _health_check_impl(self) -> None:
        """Web search specific health check."""
        # Perform a simple test search
        test_results = await self.search("test", max_results=1)
        if not isinstance(test_results, list):
            raise Exception("Health check failed: invalid response format")


# Example usage and testing
if __name__ == "__main__":
    async def test_websearch_server():
        # Create test server (using DuckDuckGo as it doesn't require API key)
        server = WebSearchServer(
            provider="duckduckgo",
            max_results=10,
            safe_search=True,
            enable_caching=True
        )
        
        async with server:
            # Test web search
            results = await server.search("Python programming tutorials")
            print(f"Search results: {len(results)} found")
            for result in results[:3]:
                print(f"  - {result['title']}: {result['url']}")
            
            # Test cached search (should be faster)
            cached_results = await server.search("Python programming tutorials")
            print(f"Cached results: {len(cached_results)} found")
            
            # Test news search
            news_results = await server.search_news("artificial intelligence", max_results=5)
            print(f"News results: {len(news_results)} found")
    
    # Run test
    # asyncio.run(test_websearch_server())
