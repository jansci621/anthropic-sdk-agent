"""Web tools: fetch URLs and search the web for real-time information."""

import ipaddress
import json
import re
import socket
import urllib.request
import urllib.parse
import urllib.error


# ── Tool Definitions (Anthropic API format) ─────────────────────────────────

WEB_TOOLS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch the content of a URL and return the text. "
            "Use this to retrieve web pages, API results, or any online resource. "
            "Returns up to 20000 characters of content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web using a search engine and return top results. "
            "Use this when you need real-time information like weather, news, "
            "stock prices, or any current data not in your training set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

# ── URL Safety ───────────────────────────────────────────────────────────────

_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> None:
    """Raise ValueError if the URL targets a private/internal address (SSRF guard)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Disallowed URL scheme: '{parsed.scheme}'. Only http/https are allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname.")

    # Resolve to IP and check for private/loopback ranges
    try:
        ip_str = socket.getaddrinfo(hostname, None)[0][4][0]
        ip = ipaddress.ip_address(ip_str)
    except (socket.gaierror, ValueError):
        # If DNS fails, let urlopen handle the error naturally
        return

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValueError(
            f"Blocked: '{hostname}' resolves to a private/internal address ({ip}). "
            "SSRF protection prevents fetching internal network resources."
        )


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json,text/plain,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _fetch(url: str, timeout: int = 15) -> str:
    """Fetch a URL and return decoded text (up to 20000 chars)."""
    _validate_url(url)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(512_000)  # read at most 512KB
        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            text = raw.decode(charset)
        except (UnicodeDecodeError, LookupError):
            text = raw.decode("utf-8", errors="replace")
    # Strip HTML tags for cleaner output
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:20000]


def _duckduckgo_search(query: str) -> list[dict]:
    """Search using DuckDuckGo HTML and parse results."""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    results = []
    # Parse result blocks from DDG HTML
    blocks = re.findall(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.S,
    )
    for link, title, snippet in blocks[:8]:
        title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        results.append({"title": title, "url": link, "snippet": snippet})

    return results


# ── Tool Dispatch ────────────────────────────────────────────────────────────

def handle_web_tool(name: str, tool_input: dict) -> str:
    """Dispatch a web tool call and return the JSON result string."""
    if name == "fetch_url":
        url = tool_input["url"]
        try:
            content = _fetch(url)
            return json.dumps(
                {"status": "ok", "url": url, "content": content},
                ensure_ascii=False,
            )
        except urllib.error.HTTPError as e:
            return json.dumps(
                {"status": "error", "url": url, "error": f"HTTP {e.code}: {e.reason}"},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {"status": "error", "url": url, "error": str(e)},
                ensure_ascii=False,
            )

    if name == "web_search":
        query = tool_input["query"]
        try:
            results = _duckduckgo_search(query)
            if not results:
                return json.dumps(
                    {"status": "ok", "query": query, "results": [], "message": "No results found"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {"status": "ok", "query": query, "results": results},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {"status": "error", "query": query, "error": str(e)},
                ensure_ascii=False,
            )

    return json.dumps({"error": f"Unknown web tool: {name}"})
