"""Small, dependency-free web search helper for EIM Chat.

Uses public HTML search endpoints (DuckDuckGo first, Bing RSS as a fallback).
Search snippets are untrusted evidence; callers should never execute or follow
instructions embedded in them. Network requests are bounded by timeout/size.
"""
from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any

_USER_AGENT = "Mozilla/5.0 (compatible; EIMChat/1.0; +https://huggingface.co/)"
_MAX_BYTES = 1_500_000


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None
        self.mode = ""

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        values = dict(attrs)
        return set((values.get("class") or "").split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = self._classes(attrs)
        attr_map = dict(attrs)
        if tag == "a" and "result__a" in classes:
            self._finish_current()
            url = _unwrap_search_url(attr_map.get("href", ""))
            if url:
                self.current = {"title": "", "url": url, "snippet": ""}
                self.mode = "title"
        elif tag in {"div", "a", "td"} and "result__snippet" in classes and self.current:
            self.mode = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.mode in {"title", "snippet"}:
            self.mode = ""
        elif tag in {"div", "td"} and self.mode == "snippet":
            self.mode = ""

    def handle_data(self, data: str) -> None:
        if not self.current or not self.mode:
            return
        value = re.sub(r"\s+", " ", data).strip()
        if value:
            self.current[self.mode] = (self.current.get(self.mode, "") + " " + value).strip()

    def _finish_current(self) -> None:
        if self.current and self.current.get("title"):
            self.results.append(self.current)
        self.current = None
        self.mode = ""

    def close(self) -> None:
        super().close()
        self._finish_current()


def _unwrap_search_url(raw_url: str) -> str:
    raw_url = html.unescape((raw_url or "").strip())
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urllib.parse.urlsplit(raw_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query = urllib.parse.parse_qs(parsed.query)
        raw_url = (query.get("uddg") or [""])[0]
        parsed = urllib.parse.urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    if parsed.username or parsed.password:
        return ""
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _clean(value: Any, limit: int = 600) -> str:
    value = html.unescape(re.sub(r"<[^>]*>", " ", str(value or "")))
    return re.sub(r"\s+", " ", value).strip()[:limit]


def parse_duckduckgo_html(source: str, limit: int = 5) -> list[dict[str, str]]:
    parser = _DuckDuckGoParser()
    parser.feed(source or "")
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in parser.results:
        url = _unwrap_search_url(row.get("url", ""))
        title = _clean(row.get("title"), 240)
        if not url or not title or url in seen:
            continue
        seen.add(url)
        unique.append({"title": title, "url": url, "snippet": _clean(row.get("snippet"), 700)})
        if len(unique) >= max(1, min(10, int(limit))):
            break
    return unique


def _fetch(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/rss+xml,application/xml;q=0.9,*/*;q=0.8"})
    with urllib.request.urlopen(request, timeout=max(2.0, min(12.0, float(timeout)))) as response:
        data = response.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        data = data[:_MAX_BYTES]
    return data


def _bing_rss(query: str, limit: int, timeout: float) -> list[dict[str, str]]:
    url = "https://www.bing.com/search?format=rss&q=" + urllib.parse.quote_plus(query)
    data = _fetch(url, timeout)
    root = ET.fromstring(data)
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in root.findall(".//item"):
        title = _clean(item.findtext("title"), 240)
        link = _unwrap_search_url(item.findtext("link", ""))
        snippet = _clean(item.findtext("description"), 700)
        if title and link and link not in seen:
            seen.add(link)
            results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= max(1, min(10, int(limit))):
            break
    return results


def search_web(query: str, limit: int = 5, timeout: float | None = None) -> list[dict[str, str]]:
    """Search the public web without third-party Python dependencies.

    Raises RuntimeError only when both public endpoints fail; a valid empty
    result list means the endpoints responded but did not return readable hits.
    """
    query = re.sub(r"\s+", " ", (query or "")).strip()
    if not query:
        return []
    timeout = timeout if timeout is not None else float(__import__("os").environ.get("EIM_WEB_TIMEOUT", "6"))
    limit = max(1, min(10, int(limit)))
    errors: list[str] = []
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
        results = parse_duckduckgo_html(_fetch(url, timeout).decode("utf-8", "replace"), limit)
        if results:
            return results
        errors.append("DuckDuckGo returned no readable results")
    except Exception as exc:
        errors.append(f"DuckDuckGo {type(exc).__name__}")
    try:
        results = _bing_rss(query, limit, timeout)
        if results:
            return results
        errors.append("Bing returned no readable results")
    except Exception as exc:
        errors.append(f"Bing {type(exc).__name__}")
    if errors and all("no readable results" not in e for e in errors):
        raise RuntimeError("Web search unavailable: " + "; ".join(errors))
    return []


def should_search_web(query: str) -> bool:
    """Only trigger automatic lookup for explicit search/current-information intent."""
    text = (query or "").casefold()
    patterns = (
        r"\b(search the web|web search|browse the web|look up online|search online|from the internet|latest|current|today|yesterday|this week|breaking news|news today|weather|exchange rate|stock price)\b",
        r"(ابحث|ابحثلي|دورلي|دور لي|تصفح الويب|بحث الويب|بحث بالنت|من النت|على الانترنت|عبر الانترنت|آخر الأخبار|اخر الاخبار|الأخبار اليوم|الاخبار اليوم|اليوم|الآن|حالياً|حاليا|أحدث|احدث|السعر الحالي|سعر اليوم)",
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)
