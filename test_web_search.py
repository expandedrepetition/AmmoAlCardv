"""Offline checks for search-intent detection and HTML result parsing."""
from web_search import parse_duckduckgo_html, should_search_web


def main() -> None:
    fixture = '''
    <div class="result results_links results_links_deep">
      <a class="result__a" href="https://example.com/article"><b>Example</b> title</a>
      <a class="result__snippet">A useful snippet with <b>details</b>.</a>
    </div>
    <div class="result results_links results_links_deep">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fnews&rut=abc">Second source</a>
      <div class="result__snippet">Current news summary.</div>
    </div>
    <div class="result"><a class="result__a" href="javascript:alert(1)">Bad URL</a></div>
    '''
    results = parse_duckduckgo_html(fixture, limit=5)
    assert len(results) == 2, results
    assert results[0]["title"] == "Example title", results
    assert results[0]["url"] == "https://example.com/article", results
    assert "useful snippet" in results[0]["snippet"], results
    assert results[1]["url"] == "https://example.org/news", results
    assert should_search_web("ابحث في الويب عن آخر الأخبار")
    assert should_search_web("Please search the web for the current price")
    assert should_search_web("شنو سعر اليوم؟")
    assert not should_search_web("write a search algorithm in Python")
    assert not should_search_web("explain recursion")
    print("PASS: web search parser and intent checks (10 assertions).")


if __name__ == "__main__":
    main()
