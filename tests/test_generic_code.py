"""Tests for generic code compressor (JS/TS, C++, SQL, HTML)."""
from tokenmin.models import CompressionLevel
from tokenmin.strategies.code import GenericCodeCompressor


def test_javascript_comments_and_strings():
    js_code = """
    // Single line comment
    function fetchUserData(userId) {
        /* Multi line
           comment block */
        const url = "https://api.example.com/user?id=" + userId; // inline comment
        const regexStr = 'keep // this string';
        const template = `template with /* not a comment */ here`;
        return fetch(url);
    }
    """
    compressor = GenericCodeCompressor()
    result = compressor.compress(js_code, lang="javascript")

    assert "Single line comment" not in result.text
    assert "Multi line" not in result.text
    assert "https://api.example.com/user?id=" in result.text
    assert "keep // this string" in result.text
    assert "template with /* not a comment */ here" in result.text
    assert result.metrics.saved_tokens > 0


def test_sql_comments():
    sql = """
    -- Select active users
    SELECT id, email /* primary contact */
    FROM users
    WHERE status = 'active -- keep this string';
    """
    compressor = GenericCodeCompressor()
    result = compressor.compress(sql, lang="sql")

    assert "Select active users" not in result.text
    assert "primary contact" not in result.text
    assert "status = 'active -- keep this string'" in result.text


def test_html_comments():
    html = """
    <!-- Header component -->
    <div class="header">
        <h1>Welcome</h1>
        <!-- Navigation bar -->
        <nav><a href="/home">Home</a></nav>
    </div>
    """
    compressor = GenericCodeCompressor()
    result = compressor.compress(html, lang="html")

    assert "<!-- Header component -->" not in result.text
    assert "<!-- Navigation bar -->" not in result.text
    assert "<h1>Welcome</h1>" in result.text
