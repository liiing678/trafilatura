"""
Unit tests for sitemaps parsing.
"""

import gzip
import logging
import os
import sys
from configparser import ConfigParser
from unittest.mock import call, patch

import pytest
from courlan import get_hostinfo

import trafilatura
from trafilatura import downloads, sitemaps
from trafilatura.checkpoint import CheckpointFormatError, CheckpointMismatchError, SitemapCheckpoint
from trafilatura.deduplication import is_similar_domain
from trafilatura.settings import DEFAULT_CONFIG
from trafilatura.utils import Response, decode_file

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

pytestmark = pytest.mark.usefixtures("mock_network")

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
RESOURCES_DIR = os.path.join(TEST_DIR, "resources")


def test_search():
    """Test search for sitemaps"""
    assert not sitemaps.sitemap_search("12345")
    assert not sitemaps.sitemap_search("12345.xml.gz")
    assert not sitemaps.sitemap_search("https://1.net/sitemap.xml.gz")
    assert not sitemaps.sitemap_search("https://bogusdomain.net/")


@pytest.mark.parametrize(("nested", "plain_text"), [(False, False), (True, False), (False, True)])
def test_search_returns_sorted_unique_urls(monkeypatch, nested, plain_text):
    root = "https://example.org/sitemap.xml" if nested or not plain_text else "https://example.org/sitemap"
    first = ["https://example.org/b", "https://example.org/a", "https://example.org/b"]
    second = ["https://example.org/b", "https://example.org/c"]

    def content(urls):
        if plain_text:
            return "\n".join(urls)
        return "<urlset>" + "".join(f"<url><loc>{url}</loc></url>" for url in urls) + "</urlset>"

    pages = {root: content(first)}
    expected = ["https://example.org/a", "https://example.org/b"]
    if nested:
        first_url = "https://example.org/sitemap-a.xml"
        second_url = "https://example.org/sitemap-b.xml"
        pages = {
            root: (
                "<sitemapindex>"
                f"<sitemap><loc>{first_url}</loc></sitemap>"
                f"<sitemap><loc>{second_url}</loc></sitemap>"
                "</sitemapindex>"
            ),
            first_url: content(first),
            second_url: content(second),
        }
        expected.append("https://example.org/c")
    monkeypatch.setattr(sitemaps, "fetch_url", lambda url, **kwargs: pages.get(url))
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)

    assert sitemaps.sitemap_search(root) == expected


@pytest.mark.parametrize(
    ("size", "config_max", "expected"),
    [
        # within the sitemap size limit, above the general download cap
        (7_500, None, 100),
        # above the sitemap size limit: still rejected
        (15_000, None, 0),
        # a lower configured cap is raised to the sitemap limit
        (7_500, 1_000, 100),
        # a higher configured cap is kept
        (15_000, 20_000, 100),
    ],
)
def test_sitemap_size_limit(monkeypatch, size, config_max, expected):
    """Sitemaps up to MAX_SITEMAP_SIZE uncompressed are read, larger ones are rejected."""
    url = "https://example.org/sitemap.xml.gz"
    # shrink the limits to a few KB so the test stays small and fast:
    # general download cap (MAX_FILE_SIZE) 5 KB, sitemap limit 10 KB
    monkeypatch.setattr(sitemaps, "MAX_SITEMAP_SIZE", 10_000)
    config = ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.set("DEFAULT", "MAX_FILE_SIZE", str(config_max if config_max is not None else 5_000))
    # 100 URLs padded to reach roughly the target uncompressed size
    padding = max((size - 100 * 45) // 100, 0)
    xml = (
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(f"<url><loc>https://example.org/{i}-{'a' * padding}</loc></url>" for i in range(100))
        + "</urlset>"
    )
    assert len(xml.encode()) >= size
    payload = gzip.compress(xml.encode())
    monkeypatch.setattr(downloads, "HAS_PYCURL", False)
    monkeypatch.setattr(downloads, "_send_urllib_request", lambda u, no_ssl, cfg: Response(payload, 200, u))
    monkeypatch.setattr(sitemaps, "is_live_page", lambda u, config: True)
    assert len(sitemaps.sitemap_search(url, config=config)) == expected
    # the caller's config is never mutated by the raised limit
    assert config.getint("DEFAULT", "MAX_FILE_SIZE") == (config_max if config_max is not None else 5_000)
    assert DEFAULT_CONFIG.getint("DEFAULT", "MAX_FILE_SIZE") == 20000000


@pytest.mark.parametrize("namespace", ["", ' xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'])
@pytest.mark.parametrize("index", [False, True])
def test_location_kind_comes_from_xml_structure(monkeypatch, namespace, index):
    root = "https://example.org/sitemap.xml"
    page = "https://example.org/article.xml"
    nested = "https://example.org/custom-map"
    page_map = f"<urlset{namespace}><url><loc>{page}</loc></url></urlset>"
    if index:
        pages = {
            root: f"<sitemapindex{namespace}><sitemap><loc>{nested}</loc></sitemap></sitemapindex>",
            nested: page_map,
        }
    else:
        pages = {root: page_map}
    fetched = []

    def fetch(url, **kwargs):
        fetched.append(url)
        return pages.get(url)

    monkeypatch.setattr(sitemaps, "fetch_url", fetch)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)

    assert sitemaps.sitemap_search(root) == [page]
    assert fetched == ([root, nested] if index else [root])


def test_parse_sitemap_without_fetching(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Parsing supplied sitemap text must not make network requests")

    monkeypatch.setattr(sitemaps, "fetch_url", unexpected_request)
    monkeypatch.setattr(sitemaps, "is_live_page", unexpected_request)
    url = "https://example.org/sitemap.xml"
    content = (
        "\ufeff \n<urlset>"
        "<url><loc>https://example.org/en/a</loc></url>"
        "<url><loc>https://example.org/de/b</loc></url>"
        "<url><loc>https://example.org/en/a</loc></url>"
        "<url><loc>https://otherdomain.net/en/c</loc></url></urlset>"
    )
    assert sitemaps.parse_sitemap(content, url) == ["https://example.org/de/b", "https://example.org/en/a"]
    assert sitemaps.parse_sitemap(content, url, target_lang="en", external=True) == [
        "https://example.org/en/a",
        "https://otherdomain.net/en/c",
    ]
    index = "<sitemapindex><sitemap><loc>https://example.org/part.xml</loc></sitemap></sitemapindex>"
    assert sitemaps.parse_sitemap(index, url) == ["https://example.org/part.xml"]
    assert sitemaps.parse_sitemap("https://example.org/a", "https://example.org/sitemap.txt") == ["https://example.org/a"]


@pytest.mark.parametrize("prefix", ["\n", "  \r\n", "\ufeff", "\ufeff\n"])
def test_leading_whitespace_before_sitemap(monkeypatch, prefix):
    url = "https://example.org/sitemap.xml"
    content = (
        prefix
        + '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.org/a</loc></url></urlset>'
    )
    assert sitemaps.is_plausible_sitemap(url, content) is True
    assert sitemaps.is_plausible_sitemap(url, prefix + "<!DOCTYPE html><html><body/></html>") is False

    monkeypatch.setattr(sitemaps, "fetch_url", lambda u, **kwargs: content if u == url else None)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda u, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    assert sitemaps.sitemap_search(url) == ["https://example.org/a"]


def test_plain_text_sitemap_html_and_quotes():
    # an HTML page whose <html> tag comes after a long comment is not a TXT sitemap
    html = "<!DOCTYPE html>\n<!-- " + "x" * 150 + " -->\n<html><body><a href='https://example.org/page'>Page</a></body></html>"
    assert sitemaps.is_plausible_sitemap("https://example.org/sitemap", html) is False
    assert sitemaps.is_plausible_sitemap("https://example.org/sitemap", html.lower()) is False
    assert sitemaps.parse_sitemap(html, "https://example.org/sitemap") == []
    # closing single quotes are not part of the link, inner apostrophes are kept
    url = "https://example.org/sitemap.txt"
    assert sitemaps.parse_sitemap("'https://example.org/page'", url) == ["https://example.org/page"]
    assert sitemaps.parse_sitemap("https://example.org/it's-here\n", url) == ["https://example.org/it%27s-here"]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/sitemap.xml?page=1",
        "https://example.org/sitemap.xml/",
        "https://example.org/sitemap/index",
        "https://example.org/sitemap.txt",
    ],
)
def test_explicit_sitemap_urls(monkeypatch, url):
    if url.endswith(".txt"):
        content = "https://example.org/a\n"
    else:
        content = (
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.org/a</loc></url></urlset>'
        )
    monkeypatch.setattr(sitemaps, "fetch_url", lambda u, **kwargs: content if u == url else None)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda u, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    assert sitemaps.sitemap_search(url) == ["https://example.org/a"]


def test_parse_sitemap_invalid_input():
    url = "https://example.org/sitemap.xml"
    with pytest.raises(TypeError, match="content must be a string"):
        sitemaps.parse_sitemap(b"<urlset/>", url)
    assert sitemaps.parse_sitemap("<urlset><url><loc>https://example.org/a</loc></url></urlset>", "invalid-url") == []
    assert sitemaps.parse_sitemap("<urlset>\ud800</urlset>", url) == []


def test_extraction():
    """Test simple link extraction"""
    # link handling
    url, domain, baseurl = "https://www.sitemaps.org/sitemap.xml", "sitemaps.org", "https://www.sitemaps.org"
    sitemap = sitemaps.SitemapObject(baseurl, domain, [])
    sitemap.handle_link(url)
    assert len(sitemap.sitemap_urls) == 1
    assert not sitemap.urls

    # same URL
    url, domain, baseurl = "https://www.sitemaps.org/sitemap.xml", "sitemaps.org", "https://www.sitemaps.org"
    sitemap = sitemaps.SitemapObject(baseurl, domain, [url])
    sitemap.current_url = url
    sitemap.handle_link(url)
    assert len(sitemap.sitemap_urls) == 1
    assert not sitemap.urls

    # malformed link
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", ["https://example.org/sitemap.xml"])
    sitemap.handle_link("http://[::1")
    assert len(sitemap.sitemap_urls) == 1
    assert not sitemap.urls

    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", ["https://example.org/sitemap.xml"])
    sitemap.handle_link("https://mydomain")
    assert len(sitemap.sitemap_urls) == 1
    assert not sitemap.urls

    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", ["https://example.org/sitemap.xml"])
    sitemap.handle_link("https://mydomain.wordpress.com/1")
    assert len(sitemap.sitemap_urls) == 1
    assert sitemap.urls == ["https://mydomain.wordpress.com/1"]

    sitemap = sitemaps.SitemapObject("https://programtalk.com", "programtalk.com", ["https://programtalk.com/sitemap.xml"])
    sitemap.handle_link(
        "http://programtalk.com/java-api-usage-examples/org.apache.xml.security.stax.securityEvent.SecurityEvent"
    )
    assert len(sitemap.sitemap_urls) == 1
    assert sitemap.urls == [
        "http://programtalk.com/java-api-usage-examples/org.apache.xml.security.stax.securityEvent.SecurityEvent"
    ]

    # similar domain names
    assert not is_similar_domain("kleins-weindepot.de", "eurosoft.net")
    assert is_similar_domain("kleins-weindepot.de", "weindepot.info")
    assert is_similar_domain("airport-frankfurt.de", "frankfurt-airport.com")

    # subdomain vs. domain: de.sitemaps.org / sitemaps.org
    url = "https://de.sitemaps.org/1"
    sitemap_url = "https://de.sitemaps.org/sitemap.xml"
    domain, baseurl = get_hostinfo(sitemap_url)
    sitemap = sitemaps.SitemapObject(baseurl, domain, [])
    sitemap.handle_link(url)
    assert not sitemap.sitemap_urls
    assert sitemap.urls == [url]

    # diverging domains
    url = "https://www.software.info/1"
    sitemap_urls = ["https://example.org/sitemap.xml"]
    domain, baseurl = get_hostinfo(sitemap_urls[0])
    sitemap = sitemaps.SitemapObject(baseurl, domain, sitemap_urls)
    sitemap.handle_link(url)
    assert len(sitemap.sitemap_urls) == 1
    assert not sitemap.urls

    # don't take this one?
    # url = 'https://subdomain.sitemaps.org/1'
    # sitemap_url = 'https://www.sitemaps.org/sitemap.xml'
    # domain, baseurl = get_hostinfo(sitemap_url)
    # sitemap.handle_link(url)  #  (url, '0')

    # safety belts
    assert sitemaps.is_plausible_sitemap("http://example.org/sitemap.xml.gz", None) is False
    assert sitemaps.is_plausible_sitemap("http://example.org/sitemap.xml.gz", b"\x1f\x8bABC") is False
    assert sitemaps.is_plausible_sitemap("http://example.org/sitemap.xml", "ABC") is False
    assert sitemaps.is_plausible_sitemap("http://test.org/sitemap.xml", "<!DOCTYPE html><html><body/></html>") is False
    assert sitemaps.is_plausible_sitemap("http://test.org/sitemap", "<!DOCTYPE html><html><body/></html>") is False
    # invalid
    sitemap = sitemaps.SitemapObject(baseurl, domain, [])
    sitemap.content = "<html>\n</html>"
    sitemap.extract_sitemap_links()
    assert not sitemap.sitemap_urls
    assert not sitemap.urls

    # parsing a file
    url, domain, baseurl = "http://www.sitemaps.org/sitemap.xml", "sitemaps.org", "http://www.sitemaps.org"
    filepath = os.path.join(RESOURCES_DIR, "sitemap.xml")
    with open(filepath, encoding="utf-8") as f:
        teststring = f.read()
    assert sitemaps.is_plausible_sitemap("http://sitemaps.org/sitemap.xml", teststring) is True
    sitemap = sitemaps.SitemapObject(baseurl, domain, [])
    sitemap.content = teststring
    sitemap.extract_sitemap_links()
    assert not sitemap.sitemap_urls
    assert len(sitemap.urls) == 84
    # hreflang
    sitemap.urls = []
    sitemap.extract_sitemap_langlinks()
    assert not sitemap.sitemap_urls
    assert not sitemap.urls

    # nested sitemaps
    url, domain, baseurl = "http://www.example.com/sitemap.xml", "example.com", "http://www.example.com"
    filepath = os.path.join(RESOURCES_DIR, "sitemap2.xml")
    with open(filepath, encoding="utf-8") as f:
        teststring = f.read()
    sitemap = sitemaps.SitemapObject(baseurl, domain, [url])
    sitemap.content = teststring
    sitemap.extract_sitemap_links()
    assert sitemap.sitemap_urls == [
        "http://www.example.com/sitemap.xml",
        "http://www.example.com/sitemap1.xml.gz",
        "http://www.example.com/sitemap2.xml.gz",
    ]
    assert not sitemap.urls

    # hreflang
    sitemap = sitemaps.SitemapObject("https://test.org/", "test.org", [], "en")
    sitemap.content = (
        '<?xml version="1.0" encoding="UTF-8"?><urlset><url><loc>http://www.test.org/english/page.html</loc></url></urlset>'
    )
    sitemap.process()
    assert (sitemap.sitemap_urls, sitemap.urls) == ([], ["http://www.test.org/english/page.html"])
    filepath = os.path.join(RESOURCES_DIR, "sitemap-hreflang.xml")
    with open(filepath, encoding="utf-8") as f:
        teststring = f.read()
    sitemap = sitemaps.SitemapObject(baseurl, domain, [], "de")
    sitemap.content = teststring
    sitemap.extract_sitemap_langlinks()
    assert sitemap.sitemap_urls == ["http://www.example.com/sitemap-de.xml.gz"]
    assert len(sitemap.urls) > 0

    # target_lang is escaped: a regex-hostile value stays inert
    sitemap = sitemaps.SitemapObject(baseurl, domain, [], "de)|(.")
    sitemap.content = teststring
    sitemap.extract_sitemap_langlinks()
    assert not sitemap.sitemap_urls
    assert not sitemap.urls

    # GZ-compressed sitemaps
    url, domain, baseurl = "https://www.sitemaps.org/sitemap.xml", "sitemaps.org", "https://www.sitemaps.org"
    filepath = os.path.join(RESOURCES_DIR, "sitemap.xml.gz")
    with open(filepath, "rb") as f:
        teststring = f.read()
    teststring = decode_file(teststring)
    assert sitemaps.is_plausible_sitemap("http://example.org/sitemap.xml.gz", teststring) is True
    sitemap = sitemaps.SitemapObject(baseurl, domain, [url])
    sitemap.content = teststring
    sitemap.extract_sitemap_links()
    assert len(sitemap.sitemap_urls) == 1
    assert len(sitemap.urls) == 84

    # check contents
    assert sitemaps.is_plausible_sitemap("http://example.org/sitemap.xml.gz?value=1", teststring) is True

    # TXT links
    content = "Tralala\nhttps://test.org/1\nhttps://test.org/2"
    assert sitemaps.is_plausible_sitemap("http://example.org/sitemap", content) is True
    sitemap = sitemaps.SitemapObject("https://test.org/", "test.org", [])
    sitemap.content = "Tralala\nhttps://test.org/1\nhttps://test.org/2"
    sitemap.process()
    assert (sitemap.sitemap_urls, sitemap.urls) == ([], ["https://test.org/1", "https://test.org/2"])

    # TXT links + language
    sitemap = sitemaps.SitemapObject("https://test.org/", "test.org", [], "en")
    sitemap.content = "Tralala\nhttps://test.org/en/1\nhttps://test.org/en/2\nhttps://test.org/es/3"
    sitemap.process()
    assert (sitemap.sitemap_urls, sitemap.urls) == ([], ["https://test.org/en/1", "https://test.org/en/2"])

    # XML sitemap with matching hreflang links: return after the langlink pass
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [], "de")
    sitemap.current_url = "https://example.org/sitemap.xml"
    sitemap.content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">'
        "<url><loc>https://example.org/page</loc>"
        '<xhtml:link rel="alternate" hreflang="de" href="https://example.org/de/page"/>'
        "</url></urlset>"
    )
    sitemap.process()
    assert sitemap.urls == ["https://example.org/de/page"]


def test_robotstxt():
    """Check if sitemaps can be found over robots.txt"""
    assert not sitemaps.find_robots_sitemaps("https://http.org")
    baseurl = "https://httpbun.com"
    assert not sitemaps.find_robots_sitemaps(baseurl)
    assert not sitemaps.extract_robots_sitemaps("# test", baseurl)
    assert not sitemaps.extract_robots_sitemaps("# test" * 10000, baseurl)
    assert sitemaps.extract_robots_sitemaps("sitemap: https://example.org/sitemap.xml", baseurl) == [
        "https://example.org/sitemap.xml"
    ]
    assert sitemaps.extract_robots_sitemaps("sitemap: http://[::1\nsitemap: /sitemap.xml", baseurl) == [
        "https://httpbun.com/sitemap.xml"
    ]
    # robots.txt files above 10000 characters are still read
    long_robots = "User-agent: *\n" + "Disallow: /private/\n" * 600 + "Sitemap: https://example.org/sitemap.xml\n"
    assert len(long_robots) > 10000
    assert sitemaps.extract_robots_sitemaps(long_robots, baseurl) == ["https://example.org/sitemap.xml"]


def test_sitemap_index_with_target_lang():
    "Regression: with target_lang, sitemaps processed while others are queued must keep their links."
    pages = {
        "https://example.org/robots.txt": "Sitemap: https://example.org/sitemap_index.xml\n",
        "https://example.org/sitemap_index.xml": (
            '<?xml version="1.0" encoding="UTF-8"?><sitemapindex>'
            "<sitemap><loc>https://example.org/sub1.xml</loc></sitemap>"
            "<sitemap><loc>https://example.org/sub2.xml</loc></sitemap>"
            "</sitemapindex>"
        ),
        "https://example.org/sub1.xml": (
            '<?xml version="1.0" encoding="UTF-8"?><urlset>'
            "<url><loc>https://example.org/de/a1.html</loc></url>"
            "<url><loc>https://example.org/de/a2.html</loc></url></urlset>"
        ),
        "https://example.org/sub2.xml": (
            '<?xml version="1.0" encoding="UTF-8"?><urlset>'
            "<url><loc>https://example.org/de/b1.html</loc></url>"
            "<url><loc>https://example.org/de/b2.html</loc></url></urlset>"
        ),
    }
    expected = [
        "https://example.org/de/a1.html",
        "https://example.org/de/a2.html",
        "https://example.org/de/b1.html",
        "https://example.org/de/b2.html",
    ]
    with (
        patch.object(sitemaps, "fetch_url", lambda url, *a, **kw: pages.get(url)),
        patch.object(sitemaps, "is_live_page", lambda u, c: True),
        patch.object(sitemaps, "sleep", lambda s: None),
    ):
        assert sorted(sitemaps.sitemap_search("https://example.org")) == expected
        assert sorted(sitemaps.sitemap_search("https://example.org", target_lang="de")) == expected


@pytest.mark.parametrize("search_url", ["https://example.org/blog", "http://example.org/blog", "https://www.example.org/blog"])
@pytest.mark.parametrize("sitemap_host", ["https://example.org", "https://www.example.org", "http://www.example.org"])
def test_subpage_filter_ignores_scheme_and_www(monkeypatch, search_url, sitemap_host):
    "regression #1011: the subpage filter must not depend on how the sitemap spells scheme and host"
    wanted = [f"{sitemap_host}/blog/a", f"{sitemap_host}/blog/sub/b"]
    body = "<urlset>" + "".join(f"<url><loc>{u}</loc></url>" for u in [*wanted, f"{sitemap_host}/news/c"]) + "</urlset>"
    # no robots.txt, the standard sitemap location is tried
    monkeypatch.setattr(sitemaps, "fetch_url", lambda url, **kwargs: body if url.endswith("/sitemap.xml") else None)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    assert sitemaps.sitemap_search(search_url) == wanted


def test_subpage_filter_keeps_other_hosts_out(monkeypatch):
    "With external links allowed, only the requested host and path are kept."
    urls = [
        "https://www.example.org/blog/a",
        "https://other.org/blog/b",
        "https://notexample.org/blog/c",
        "https://blog.example.org/blog/d",
    ]
    body = "<urlset>" + "".join(f"<url><loc>{u}</loc></url>" for u in urls) + "</urlset>"
    monkeypatch.setattr(sitemaps, "fetch_url", lambda url, **kwargs: body if url.endswith("/sitemap.xml") else None)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    assert sitemaps.sitemap_search("https://example.org/blog", external=True) == ["https://www.example.org/blog/a"]


def test_lang_filter_before_cleaning():
    "Language-root URLs with trailing slashes must be filtered even if cleaning strips the slash."
    sitemap = sitemaps.SitemapObject("https://www.sitemaps.org", "sitemaps.org", [], target_lang="de")
    sitemap.handle_link("https://www.sitemaps.org/da/")
    sitemap.handle_link("https://www.sitemaps.org/da/faq.html")
    assert not sitemap.urls
    sitemap.handle_link("https://www.sitemaps.org/de/faq.html")
    assert sitemap.urls == ["https://www.sitemaps.org/de/faq.html"]


def test_guess_order():
    "Without robots.txt hints, the most common sitemap location must be tried first."
    fetched = []

    def fake_fetch(url, *a, **kw):
        fetched.append(url)

    with (
        patch.object(sitemaps, "fetch_url", fake_fetch),
        patch.object(sitemaps, "is_live_page", lambda u, c: True),
        patch.object(sitemaps, "sleep", lambda s: None),
    ):
        sitemaps.sitemap_search("https://example.org")
    assert fetched[0] == "https://example.org/robots.txt"
    assert fetched[1] == "https://example.org/sitemap.xml"


@pytest.mark.parametrize(
    ("start", "maps", "fetched", "expected"),
    [
        ("", {"sitemap.xml": ["a"], "sitemap.xml.gz": ["b"]}, ["sitemap.xml"], ["a"]),
        ("", {"sitemap.xml.gz": ["b"]}, ["sitemap.xml", "sitemap.xml.gz"], ["b"]),
        (
            "",
            {"sitemap.xml": ["part-a.xml", "part-b.xml"], "part-a.xml": ["a"], "part-b.xml": ["b"]},
            ["sitemap.xml", "part-b.xml", "part-a.xml"],
            ["a", "b"],
        ),
        (
            "",
            {"sitemap.xml": ["sitemap.xml.gz"], "sitemap": ["a"]},
            ["sitemap.xml", "sitemap.xml.gz", "sitemap"],
            ["a"],
        ),
        (
            "/blog",
            {"sitemap.xml": ["news/a"], "sitemap.xml.gz": ["blog/b"]},
            ["sitemap.xml", "sitemap.xml.gz"],
            ["blog/b"],
        ),
    ],
)
def test_stop_guessing_after_a_sitemap_yields_links(monkeypatch, start, maps, fetched, expected):
    base = "https://example.org"

    def fetch(url, **kwargs):
        entries = maps.get(url.removeprefix(f"{base}/"))
        if not entries:
            return None
        root, tag = ("sitemapindex", "sitemap") if entries[0].endswith((".xml", ".xml.gz")) else ("urlset", "url")
        locations = "".join(f"<{tag}><loc>{base}/{entry}</loc></{tag}>" for entry in entries)
        return f"<{root}>{locations}</{root}>"

    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "find_robots_sitemaps", lambda url, config: [])
    with patch.object(sitemaps, "fetch_url", side_effect=fetch) as request, patch.object(sitemaps, "sleep") as wait:
        assert sitemaps.sitemap_search(f"{base}{start}", sleep_time=0.25) == [f"{base}/{page}" for page in expected]

    assert [args[0] for args, _ in request.call_args_list] == [f"{base}/{path}" for path in fetched]
    assert wait.call_args_list == [call(0.25)] * (len(fetched) - 1)


def test_whole():
    "Test whole process."
    results = sitemaps.sitemap_search("https://www.sitemaps.org", target_lang="de", max_sitemaps=1)
    assert len(results) == 8

    trafilatura.settings.MAX_SITEMAPS_SEEN = 1
    results = sitemaps.sitemap_search("https://www.sitemaps.org", target_lang="de")
    assert len(results) == 8


@pytest.mark.parametrize("prefix", ["", "sm:", "site-map:"])
@pytest.mark.parametrize("declaration", ["", '<?xml version="1.0" encoding="UTF-8"?>'])
@pytest.mark.parametrize("index", [False, True])
def test_sitemap_namespaces(prefix, declaration, index):
    """Namespace prefixes do not change page or nested sitemap discovery."""
    root, child = ("sitemapindex", "sitemap") if index else ("urlset", "url")
    namespace = f'xmlns{":" + prefix[:-1] if prefix else ""}="http://www.sitemaps.org/schemas/sitemap/0.9"'
    url = "https://example.org/nested.xml" if index else "https://example.org/page"
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [])
    sitemap.current_url = "https://example.org/sitemap.xml"
    sitemap.content = (
        f"{declaration}<{prefix}{root} {namespace}>"
        f"<{prefix}{child}><{prefix}loc><![CDATA[{url}]]></{prefix}loc></{prefix}{child}>"
        f"</{prefix}{root}>"
    )
    sitemap.process()
    assert (sitemap.sitemap_urls, sitemap.urls) == (([url], []) if index else ([], [url]))


def test_sitemap_namespace_scope():
    """Ignore extension locations even when a prefix is rebound locally."""
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [])
    sitemap.current_url = "https://example.org/sitemap.xml"
    sitemap.content = (
        '<s:urlset xmlns:s="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:image="urn:image">'
        "<s:url><s:loc>https://example.org/page</s:loc>"
        "<image:loc>https://example.org/image</image:loc>"
        '<s:loc xmlns:s="urn:other">https://example.org/other</s:loc>'
        "</s:url></s:urlset>"
    )
    sitemap.process()
    assert sitemap.urls == ["https://example.org/page"]


@pytest.mark.parametrize(
    "namespace",
    [
        "http://www.google.com/schemas/sitemap/0.84",
        "https://www.sitemaps.org/schemas/sitemap/0.9",
        "http://www.sitemaps.org/schemas/sitemap/0.9/",
        "http://www.sitemaps.org/schemas/sitemap/0.90",
    ],
)
@pytest.mark.parametrize("prefix", ["", "sm:"])
@pytest.mark.parametrize("index", [False, True])
def test_sitemap_namespace_variants(namespace, prefix, index):
    """Legacy, https and trailing-slash sitemap namespaces are accepted, extension locations are not."""
    root, child = ("sitemapindex", "sitemap") if index else ("urlset", "url")
    url = "https://example.org/nested.xml" if index else "https://example.org/page"
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [])
    sitemap.current_url = "https://example.org/sitemap.xml"
    sitemap.content = (
        f'<{prefix}{root} xmlns{":" + prefix[:-1] if prefix else ""}="{namespace}"'
        ' xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">'
        f"<{prefix}{child}><{prefix}loc>{url}</{prefix}loc>"
        "<image:image><image:loc>https://example.org/image.jpg</image:loc></image:image>"
        f"</{prefix}{child}></{prefix}{root}>"
    )
    sitemap.process()
    assert (sitemap.sitemap_urls, sitemap.urls) == (([url], []) if index else ([], [url]))


@pytest.mark.parametrize("location", ["&external;", "https://example.org/&external;", "https://example.org/<nested/>"])
def test_sitemap_locations_do_not_expand_entities(location):
    """Location extraction does not resolve entities or concatenate child markup."""
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [])
    sitemap.content = (
        '<!DOCTYPE urlset [<!ENTITY external SYSTEM "file:///not-a-sitemap-resource">]>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{location}</loc></url></urlset>"
    )
    sitemap.extract_sitemap_links()
    assert not sitemap.urls


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize(
    "location",
    ["\n    {url}\n  ", " {url}", "{url}\t", "\n  <![CDATA[{url}]]>\n"],
)
def test_sitemap_locations_ignore_surrounding_whitespace(location, index):
    """Pretty-printed locations are not dropped because of their indentation."""
    root, child = ("sitemapindex", "sitemap") if index else ("urlset", "url")
    url = "https://example.org/nested.xml" if index else "https://example.org/page"
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [])
    sitemap.content = (
        f'<{root} xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<{child}><loc>{location.format(url=url)}</loc></{child}></{root}>"
    )
    sitemap.extract_sitemap_links()
    assert (sitemap.sitemap_urls, sitemap.urls) == (([url], []) if index else ([], [url]))


def test_sitemap_namespace_link_limit(monkeypatch):
    """Namespaced extraction keeps the existing maximum-location bound."""
    monkeypatch.setattr(sitemaps, "MAX_LINKS", 1)
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [])
    sitemap.content = (
        '<s:urlset xmlns:s="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<s:url><s:loc>https://example.org/first?a=1&amp;b=2</s:loc></s:url>"
        "<s:url><s:loc>https://example.org/second</s:loc></s:url></s:urlset>"
    )
    sitemap.extract_sitemap_links()
    assert sitemap.urls == ["https://example.org/first?a=1&b=2"]


@pytest.mark.parametrize("content", ["", "not an XML document"])
def test_sitemap_xml_without_a_root_is_ignored(content):
    """Malformed input cannot yield a sitemap location."""
    sitemap = sitemaps.SitemapObject("https://example.org", "example.org", [])
    sitemap.content = content
    sitemap.extract_sitemap_links()
    assert sitemap.urls == []
    assert sitemap.sitemap_urls == []


def test_sleep_time_defaults_to_config(monkeypatch):
    """Without an explicit sleep_time, the wait comes from SLEEP_TIME in the config."""
    config = ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.set("DEFAULT", "SLEEP_TIME", "10")
    robots = ["https://example.org/a.xml", "https://example.org/b.xml"]
    monkeypatch.setattr(sitemaps, "find_robots_sitemaps", lambda url, config: list(robots))
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "fetch_url", lambda url, **kwargs: None)

    with patch.object(sitemaps, "sleep") as mock_sleep:
        sitemaps.sitemap_search("https://example.org", config=config)
    assert mock_sleep.call_args_list == [call(10.0)]

    with patch.object(sitemaps, "sleep") as mock_sleep:
        sitemaps.sitemap_search("https://example.org", sleep_time=0.5, config=config)
    assert mock_sleep.call_args_list == [call(0.5)]


# --------------------------------------------------------------------------- #
# resumable searches with checkpoints
# --------------------------------------------------------------------------- #

# a fixed local graph: one index, three nested sitemaps, overlapping page links
GRAPH_ROOT = "https://example.org/sitemap_index.xml"
GRAPH_NESTED = [f"https://example.org/sitemap-{letter}.xml" for letter in ("a", "b", "c")]
GRAPH_PAGES = {
    GRAPH_NESTED[0]: ["https://example.org/a1", "https://example.org/a2", "https://example.org/shared"],
    GRAPH_NESTED[1]: ["https://example.org/b1", "https://example.org/shared"],
    GRAPH_NESTED[2]: ["https://example.org/c1", "https://example.org/c2"],
}
GRAPH_EXPECTED = sorted({url for urls in GRAPH_PAGES.values() for url in urls})


def _graph_pages():
    pages = {
        GRAPH_ROOT: (
            "<sitemapindex>" + "".join(f"<sitemap><loc>{url}</loc></sitemap>" for url in GRAPH_NESTED) + "</sitemapindex>"
        ),
        "https://example.org/robots.txt": f"Sitemap: {GRAPH_ROOT}\n",
    }
    for sitemap_url, page_urls in GRAPH_PAGES.items():
        pages[sitemap_url] = "<urlset>" + "".join(f"<url><loc>{page_url}</loc></url>" for page_url in page_urls) + "</urlset>"
    return pages


def _patch_graph(monkeypatch):
    pages = _graph_pages()
    requested = []

    def fetch(url, **kwargs):
        requested.append(url)
        return pages.get(url)

    monkeypatch.setattr(sitemaps, "fetch_url", fetch)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    return requested


def test_checkpoint_budget_requires_checkpoint():
    with pytest.raises(ValueError, match="requires checkpoint_path"):
        sitemaps.sitemap_search("https://example.org/sitemap.xml", per_run_budget=2)
    with pytest.raises(ValueError, match="positive integer"):
        sitemaps.sitemap_search("https://example.org/sitemap.xml", checkpoint_path="x", per_run_budget=0)


def test_checkpoint_segmented_run_matches_one_shot(monkeypatch, tmp_path):
    """Splitting the traversal into budgeted runs yields the same link set."""
    requested = _patch_graph(monkeypatch)
    state = tmp_path / "checkpoint.json"
    kwargs = {"checkpoint_path": str(state)}

    # budget counts every newly submitted sitemap, the index is the first one
    first = sitemaps.sitemap_search(GRAPH_ROOT, per_run_budget=1, **kwargs)
    assert first == []  # the index only lists nested sitemaps, no pages
    checkpoint = SitemapCheckpoint.load(state)
    assert checkpoint.completed is False
    assert checkpoint.processed_count == 1
    assert checkpoint.queue == GRAPH_NESTED
    # already submitted sitemaps are not requested again
    assert len(requested) == len(set(requested))

    accumulated = set(first)
    for _ in range(len(GRAPH_NESTED)):
        accumulated |= set(sitemaps.sitemap_search(GRAPH_ROOT, per_run_budget=1, **kwargs))

    checkpoint = SitemapCheckpoint.load(state)
    assert checkpoint.completed is True
    assert checkpoint.processed_count == len(GRAPH_NESTED) + 1
    assert sorted(accumulated) == GRAPH_EXPECTED
    # every sitemap URL was requested at most once across processes
    assert len(requested) == len(set(requested))
    assert set(requested) >= {GRAPH_ROOT, *GRAPH_NESTED}
    # a completed checkpoint returns the final set without new requests
    before = len(requested)
    again = sitemaps.sitemap_search(GRAPH_ROOT, per_run_budget=1, **kwargs)
    assert again == GRAPH_EXPECTED
    assert len(requested) == before


def test_checkpoint_unlimited_run_completes(monkeypatch, tmp_path):
    """A checkpoint without a budget finishes in one run and marks itself complete."""
    _patch_graph(monkeypatch)
    state = tmp_path / "checkpoint.json"
    result = sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state))
    assert result == GRAPH_EXPECTED
    assert SitemapCheckpoint.load(state).completed is True


def test_checkpoint_state_survives_a_kill_between_sitemaps(monkeypatch, tmp_path):
    """A process killed after the last checkpoint write resumes without re-requesting."""
    requested = _patch_graph(monkeypatch)
    state = tmp_path / "checkpoint.json"

    sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=1)
    first_requests = list(requested)
    # simulate fresh processes: the module-level state is gone, only the file remains
    for _ in range(len(GRAPH_NESTED)):
        sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=1)

    assert SitemapCheckpoint.load(state).completed is True
    assert len(requested) == len(set(requested))
    assert GRAPH_ROOT in first_requests
    assert GRAPH_ROOT in requested


def test_checkpoint_failed_fetch_is_recorded_and_not_retried(monkeypatch, tmp_path):
    """An unreachable nested sitemap counts as processed once and is skipped on resume."""
    pages = _graph_pages()
    requested = []

    def flaky_fetch(url, **kwargs):
        requested.append(url)
        if url == GRAPH_NESTED[1]:
            return None  # download failure on sitemap-b
        return pages.get(url)

    monkeypatch.setattr(sitemaps, "fetch_url", flaky_fetch)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    state = tmp_path / "checkpoint.json"

    sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=2)
    sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=2)

    checkpoint = SitemapCheckpoint.load(state)
    assert checkpoint.completed is True
    assert GRAPH_NESTED[1] in checkpoint.seen
    # the failed sitemap was fetched exactly once
    assert requested.count(GRAPH_NESTED[1]) == 1
    # links from the two reachable nested sitemaps are still complete
    assert checkpoint.urls == sorted(set(GRAPH_PAGES[GRAPH_NESTED[0]]) | set(GRAPH_PAGES[GRAPH_NESTED[2]]))


def test_checkpoint_gzip_nested_sitemaps(monkeypatch, tmp_path):
    """Gzip-compressed nested sitemaps are supported across segments."""
    nested = "https://example.org/part.xml.gz"
    page = "https://example.org/gzip-page"
    xml = f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{page}</loc></url></urlset>'
    payloads = {
        GRAPH_ROOT: (
            f'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"<sitemap><loc>{nested}</loc></sitemap></sitemapindex>"
        ).encode(),
        nested: gzip.compress(xml.encode()),
        "https://example.org/robots.txt": f"Sitemap: {GRAPH_ROOT}\n".encode(),
    }
    requested = []

    def send(url, no_ssl, cfg):
        requested.append(url)
        payload = payloads.get(url)
        return Response(payload, 200, url) if payload is not None else None

    monkeypatch.setattr(downloads, "HAS_PYCURL", False)
    monkeypatch.setattr(downloads, "_send_urllib_request", send)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    state = tmp_path / "checkpoint.json"

    first = sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=1)
    second = sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=1)
    assert SitemapCheckpoint.load(state).completed is True
    assert sorted(set(first) | set(second)) == [page]
    assert requested.count(nested) == 1


def test_checkpoint_corrupt_state_aborts_before_requests(monkeypatch, tmp_path):
    """A corrupt checkpoint raises and the network is never touched."""
    state = tmp_path / "checkpoint.json"
    state.write_text("{broken", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        pytest.fail("no request must be sent with a corrupt checkpoint")

    monkeypatch.setattr(sitemaps, "fetch_url", fail_if_called)
    monkeypatch.setattr(sitemaps, "is_live_page", fail_if_called)
    with pytest.raises(CheckpointFormatError):
        sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state))
    # the corrupt file is not overwritten
    assert state.read_text(encoding="utf-8") == "{broken"


def test_checkpoint_policy_mismatch_aborts_before_requests(monkeypatch, tmp_path):
    """Entry, language or external-link mismatch raises before any request."""
    state = tmp_path / "checkpoint.json"
    SitemapCheckpoint.open(state, GRAPH_ROOT, target_lang="de").save()

    def fail_if_called(*args, **kwargs):
        pytest.fail("no request must be sent on a policy mismatch")

    monkeypatch.setattr(sitemaps, "fetch_url", fail_if_called)
    monkeypatch.setattr(sitemaps, "is_live_page", fail_if_called)
    with pytest.raises(CheckpointMismatchError):
        sitemaps.sitemap_search(GRAPH_ROOT, target_lang="en", checkpoint_path=str(state))
    with pytest.raises(CheckpointMismatchError):
        sitemaps.sitemap_search("https://example.org/other.xml", checkpoint_path=str(state))
    # the original state is untouched
    assert SitemapCheckpoint.load(state).target_lang == "de"


def test_checkpoint_write_failure_keeps_last_good_state(monkeypatch, tmp_path):
    """If a state write fails mid-run, the previous valid file remains usable."""
    _patch_graph(monkeypatch)
    state = tmp_path / "checkpoint.json"
    # first run writes a valid state
    sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=1)
    good_content = state.read_text(encoding="utf-8")

    real_replace = os.replace

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(CheckpointFormatError):
        sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=1)

    # last good state is intact, no temp file lingers and resumption still works
    assert state.read_text(encoding="utf-8") == good_content
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(os, "replace", real_replace)
    result = sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state))
    assert result == GRAPH_EXPECTED


def test_checkpoint_without_flag_keeps_one_shot_behavior(monkeypatch, tmp_path):
    """Without checkpoint arguments, no state file is created and output is unchanged."""
    requested = _patch_graph(monkeypatch)
    state = tmp_path / "checkpoint.json"
    result = sitemaps.sitemap_search(GRAPH_ROOT)
    assert result == GRAPH_EXPECTED
    assert not state.exists()
    # plain run still requests each sitemap once
    assert len(requested) == len(set(requested))


def test_checkpoint_respects_max_sitemaps_cap(monkeypatch, tmp_path):
    """The absolute max_sitemaps cap terminates the traversal as completed."""
    _patch_graph(monkeypatch)
    state = tmp_path / "checkpoint.json"
    for _ in range(5):
        sitemaps.sitemap_search(GRAPH_ROOT, checkpoint_path=str(state), per_run_budget=1, max_sitemaps=2)
    checkpoint = SitemapCheckpoint.load(state)
    assert checkpoint.completed is True
    assert len(checkpoint.seen) == 2


def test_checkpoint_guess_frontier_resumes(monkeypatch, tmp_path):
    """Fallback guesses discovered in one process are not re-guessed after a restart."""
    base = "https://example.org"
    pages = {
        f"{base}/sitemap.xml": None,
        f"{base}/sitemap.xml.gz": "<urlset><url><loc>https://example.org/gz</loc></url></urlset>",
    }
    requested = []

    def fetch(url, **kwargs):
        requested.append(url)
        return pages.get(url)

    monkeypatch.setattr(sitemaps, "find_robots_sitemaps", lambda url, config: [])
    monkeypatch.setattr(sitemaps, "fetch_url", fetch)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    state = tmp_path / "checkpoint.json"

    for _ in range(6):
        sitemaps.sitemap_search(f"{base}/", checkpoint_path=str(state), per_run_budget=1)
        if SitemapCheckpoint.load(state).completed:
            break

    checkpoint = SitemapCheckpoint.load(state)
    assert checkpoint.completed is True
    assert f"{base}/sitemap.xml" in checkpoint.seen
    assert f"{base}/sitemap.xml.gz" in checkpoint.seen
    assert requested.count(f"{base}/sitemap.xml") == 1
    assert requested.count(f"{base}/sitemap.xml.gz") == 1


def test_checkpoint_segmented_matches_one_shot_with_language_filter(monkeypatch, tmp_path):
    """A target_lang filter gives the same result across segments as one full run."""
    pages = _graph_pages()
    pages[GRAPH_NESTED[0]] = (
        "<urlset><url><loc>https://example.org/en/a1</loc></url><url><loc>https://example.org/de/a2</loc></url></urlset>"
    )
    requested = []

    def fetch(url, **kwargs):
        requested.append(url)
        return pages.get(url)

    monkeypatch.setattr(sitemaps, "fetch_url", fetch)
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    state = tmp_path / "checkpoint.json"

    segmented = set()
    for _ in range(6):
        segmented |= set(sitemaps.sitemap_search(GRAPH_ROOT, target_lang="en", checkpoint_path=str(state), per_run_budget=1))
        if SitemapCheckpoint.load(state).completed:
            break

    one_shot = sitemaps.sitemap_search(GRAPH_ROOT, target_lang="en")
    assert sorted(segmented) == one_shot
    # the German-only link is filtered out, identically in both executions
    assert "https://example.org/de/a2" not in segmented
    assert "https://example.org/de/a2" not in one_shot


def test_checkpoint_segmented_matches_one_shot_with_external_links(monkeypatch, tmp_path):
    """external=True keeps external links identically across segments."""
    pages = _graph_pages()
    pages[GRAPH_NESTED[2]] = (
        "<urlset><url><loc>https://example.org/c1</loc></url><url><loc>https://other-domain.net/x</loc></url></urlset>"
    )
    monkeypatch.setattr(sitemaps, "fetch_url", lambda url, **kwargs: pages.get(url))
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    state = tmp_path / "checkpoint.json"

    segmented = set()
    for _ in range(6):
        segmented |= set(sitemaps.sitemap_search(GRAPH_ROOT, external=True, checkpoint_path=str(state), per_run_budget=1))
        if SitemapCheckpoint.load(state).completed:
            break

    one_shot = sitemaps.sitemap_search(GRAPH_ROOT, external=True)
    assert sorted(segmented) == one_shot
    assert "https://other-domain.net/x" in segmented


def test_checkpoint_subpage_filter_accumulates_only_matching_links(monkeypatch, tmp_path):
    """The subpage URL filter applies to the cumulative set on every save."""
    # a non-sitemap entry under /blog activates the subpage URL filter
    entry = "https://example.org/blog"
    root = "https://example.org/sitemap.xml"
    nested = "https://example.org/part.xml"
    pages = {
        "https://example.org/robots.txt": f"Sitemap: {root}\n",
        root: f"<sitemapindex><sitemap><loc>{nested}</loc></sitemap></sitemapindex>",
        nested: (
            "<urlset>"
            "<url><loc>https://example.org/blog/post-1</loc></url>"
            "<url><loc>https://example.org/news/post-2</loc></url>"
            "</urlset>"
        ),
    }
    monkeypatch.setattr(sitemaps, "fetch_url", lambda url, **kwargs: pages.get(url))
    monkeypatch.setattr(sitemaps, "is_live_page", lambda url, config: True)
    monkeypatch.setattr(sitemaps, "sleep", lambda seconds: None)
    state = tmp_path / "checkpoint.json"

    segmented = set()
    for _ in range(6):
        segmented |= set(sitemaps.sitemap_search(entry, checkpoint_path=str(state), per_run_budget=1))
        if SitemapCheckpoint.load(state).completed:
            break

    one_shot = sitemaps.sitemap_search(entry)
    checkpoint = SitemapCheckpoint.load(state)
    assert checkpoint.completed is True
    assert one_shot == ["https://example.org/blog/post-1"]
    assert sorted(segmented) == one_shot
    assert checkpoint.urls == ["https://example.org/blog/post-1"]
