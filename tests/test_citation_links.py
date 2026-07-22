"""Unit tests for the clickable citation link functionality in the Streamlit frontend."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.frontend.streamlit_app import _build_chunk_id_to_source_map, _make_chunk_id_links


def test_build_chunk_id_to_source_map():
    """Verify mapping from chunk_id to source_id is built correctly."""
    citations = [
        {"chunk_id": "health_policy_hdfcergo_0", "source_id": "health_policy_hdfcergo"},
        {"chunk_id": "health_policy_hdfcergo_1", "source_id": "health_policy_hdfcergo"},
        {"chunk_id": "irda_health_reg_2016_2", "source_id": "irda_health_reg_2016"},
    ]
    mapping = _build_chunk_id_to_source_map(citations)
    assert mapping == {
        "health_policy_hdfcergo_0": "health_policy_hdfcergo",
        "health_policy_hdfcergo_1": "health_policy_hdfcergo",
        "irda_health_reg_2016_2": "irda_health_reg_2016",
    }


def test_build_chunk_id_to_source_map_empty():
    """Verify empty citations produce empty mapping."""
    assert _build_chunk_id_to_source_map([]) == {}
    assert _build_chunk_id_to_source_map([{"chunk_id": ""}]) == {}
    assert _build_chunk_id_to_source_map([{"source_id": "doc1"}]) == {}


def test_make_chunk_id_links_creates_anchor_tags():
    """Verify [chunk_id] references are replaced with anchor links to chunk details."""
    citations = [
        {"chunk_id": "health_policy_hdfcergo_0", "source_id": "health_policy_hdfcergo"},
    ]
    chunk_to_source = _build_chunk_id_to_source_map(citations)
    api_url = "http://127.0.0.1:8000"
    
    text = "The waiting period is 4 years according to [health_policy_hdfcergo_0]."
    result = _make_chunk_id_links(text, chunk_to_source, api_url)
    
    # Should generate an anchor link to #chunk-{chunk_id} for same-page navigation
    assert 'href="#chunk-health_policy_hdfcergo_0"' in result
    assert 'class="citation-link"' in result
    assert 'title="View source chunk: health_policy_hdfcergo_0"' in result
    assert "[health_policy_hdfcergo_0]" in result  # The text inside the link is preserved
    # Should NOT generate download links to backend
    assert "/sources/" not in result


def test_make_chunk_id_links_no_mapping():
    """Verify unknown chunk IDs are preserved as styled text (no link)."""
    chunk_to_source = {"known_chunk": "known_doc"}
    api_url = "http://127.0.0.1:8000"
    
    text = "Some [unknown_chunk] and [known_chunk] references."
    result = _make_chunk_id_links(text, chunk_to_source, api_url)
    
    # Known chunk should be a citation-link anchor pointing to its chunk ID
    assert 'href="#chunk-known_chunk"' in result
    assert 'class="citation-link"' in result
    # Unknown chunk should be rendered as a styled span (no link)
    assert 'class="citation-ref"' in result
    assert 'href="#chunk-unknown_chunk"' not in result


def test_make_chunk_id_links_no_api_url():
    """Verify no links are generated when api_url is empty."""
    chunk_to_source = {"chunk_0": "source_doc"}
    
    text = "Reference [chunk_0]."
    result = _make_chunk_id_links(text, chunk_to_source, "")
    
    # When no api_url, no anchor tag should be created
    assert "<a" not in result
    assert 'href=' not in result


def test_make_chunk_id_links_multiple_references():
    """Verify multiple chunk references are all linked with anchor tags."""
    citations = [
        {"chunk_id": "doc_a_0", "source_id": "doc_a"},
        {"chunk_id": "doc_b_1", "source_id": "doc_b"},
    ]
    chunk_to_source = _build_chunk_id_to_source_map(citations)
    api_url = "http://localhost:8000"
    
    text = "According to [doc_a_0] and [doc_b_1], the answer is yes."
    result = _make_chunk_id_links(text, chunk_to_source, api_url)
    
    # Should generate anchor links to #chunk-{chunk_id}
    assert 'href="#chunk-doc_a_0"' in result
    assert 'href="#chunk-doc_b_1"' in result
    assert "[doc_a_0]" in result
    assert "[doc_b_1]" in result


def test_make_chunk_id_links_no_brackets():
    """Verify text without any bracket references remains unchanged (except HTML escaping handled by caller)."""
    chunk_to_source = {}
    api_url = "http://localhost:8000"
    
    text = "Plain text without any citation references."
    result = _make_chunk_id_links(text, chunk_to_source, api_url)
    
    assert result == text  # Unchanged since we directly return HTML, and caller handles escaping