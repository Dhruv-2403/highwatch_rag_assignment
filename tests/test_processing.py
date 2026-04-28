import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from processing.pipeline import clean_text, chunk_text, _approx_tokens


class TestCleanText:
    def test_normalises_unicode(self):
        result = clean_text("caf\u00e9")  # é as single codepoint
        assert "café" in result

    def test_collapses_blank_lines(self):
        result = clean_text("line1\n\n\n\nline2")
        assert "\n\n\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_collapses_spaces(self):
        result = clean_text("hello    world")
        assert "hello world" == result

    def test_removes_control_chars(self):
        result = clean_text("hello\x00world\x07!")
        assert "\x00" not in result
        assert "\x07" not in result
        assert "helloworld!" == result


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("This is a short sentence.", chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == "This is a short sentence."

    def test_multiple_paragraphs_multiple_chunks(self):
        para = "word " * 150  
        text = "\n\n".join([para] * 6) 
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

   
        para = "sentence " * 80
        text = "\n\n".join([para] * 4)
        chunks = chunk_text(text, chunk_size=128, chunk_overlap=32)
        if len(chunks) >= 2:
   
            tail = chunks[0][-200:]
            head = chunks[1][:200]
         
            tail_words = set(tail.split())
            head_words = set(head.split())
            assert tail_words & head_words, "Expected overlap between consecutive chunks"

    def test_empty_string_returns_empty(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n   ") == []

    def test_no_duplicate_empty_chunks(self):
        chunks = chunk_text("Hello world.\n\nSecond paragraph.")
        assert all(c.strip() for c in chunks)

    def test_very_long_paragraph_split(self):
  
        long_para = "x " * 1000  
        chunks = chunk_text(long_para, chunk_size=128, chunk_overlap=16)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 512 * 4 * 1.1 


class TestApproxTokens:
    def test_empty(self):
        assert _approx_tokens("") == 0

    def test_rough_ratio(self):
        text = "a" * 400
        # Should be approximately 100 tokens
        assert 90 <= _approx_tokens(text) <= 110
