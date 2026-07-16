"""Tests for metadata filter fallback behavior in hybrid retrieval."""

import logging
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from app.rag.chunkers import Chunk, ChunkConfig, chunk_document
from app.rag.retriever_hybrid import hybrid_retrieve
from app.config import get_settings


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks with different insurance types.
    
    Note: The motor_policy_1 and motor_policy_2 chunks are intentionally
    marked as health insurance type to simulate a metadata mismatch scenario
    where the filter doesn't match any chunks.
    """
    chunks = [
        Chunk(
            text="This motor insurance policy covers own damage to the vehicle.",
            source_id="motor_policy_1",
            source_path="/data/motor_policy_1.pdf",
            doc_type="policy_wording",
            insurance_type="health",  # Mislabeled as health
            chunk_index=0,
            raw_metadata={"insurance_type": "health"},
        ),
        Chunk(
            text="Third party liability coverage is included.",
            source_id="motor_policy_1",
            source_path="/data/motor_policy_1.pdf",
            doc_type="policy_wording",
            insurance_type="health",  # Mislabeled as health
            chunk_index=1,
            raw_metadata={"insurance_type": "health"},
        ),
        Chunk(
            text="Day care procedures are covered under this health policy.",
            source_id="health_policy_1",
            source_path="/data/health_policy_1.pdf",
            doc_type="policy_wording",
            insurance_type="health",
            chunk_index=0,
            raw_metadata={"insurance_type": "health"},
        ),
        Chunk(
            text="Pre-existing diseases are covered after 4 years.",
            source_id="health_policy_1",
            source_path="/data/health_policy_1.pdf",
            doc_type="policy_wording",
            insurance_type="health",
            chunk_index=1,
            raw_metadata={"insurance_type": "health"},
        ),
        Chunk(
            text="Network hospital cashless claims are available.",
            source_id="health_policy_2",
            source_path="/data/health_policy_2.pdf",
            doc_type="policy_wording",
            insurance_type="health",
            chunk_index=0,
            raw_metadata={"insurance_type": "health"},
        ),
    ]
    return chunks


@pytest.fixture
def mock_settings_fallback_enabled():
    """Mock settings with fallback enabled."""
    with patch("app.config.get_settings") as mock_get_settings:
        settings = MagicMock()
        settings.RETRIEVAL_FILTER_FALLBACK_ENABLED = True
        mock_get_settings.return_value = settings
        yield settings


@pytest.fixture
def mock_settings_fallback_disabled():
    """Mock settings with fallback disabled."""
    with patch("app.config.get_settings") as mock_get_settings:
        settings = MagicMock()
        settings.RETRIEVAL_FILTER_FALLBACK_ENABLED = False
        mock_get_settings.return_value = settings
        yield settings


# ── Test Cases ────────────────────────────────────────────────────────────────


class TestMetadataFilterFallback:
    """Test metadata filter fallback behavior."""

    def test_case_1_fallback_triggered_with_motor_filter_on_health_query(
        self, sample_chunks, mock_settings_fallback_enabled, caplog
    ):
        """Case 1: Query about day care with motor filter should trigger fallback."""
        query = "What is covered under day care procedures in this policy?"
        metadata_filter = {"insurance_type": "motor"}
        
        with caplog.at_level(logging.WARNING):
            results = hybrid_retrieve(
                sample_chunks,
                query,
                k=5,
                metadata_filter=metadata_filter,
                vector_store=None,
            )
        
        # Should trigger fallback and return results
        assert len(results) > 0, "Should return results after fallback"
        
        # Check that fallback was logged
        assert any("Metadata filter returned zero chunks, applying fallback" in record.message 
                  for record in caplog.records)
        
        # Check that results have fallback metadata
        for result in results:
            assert result.get("fallback_used") is True
            assert result.get("original_metadata_filter") == metadata_filter
            assert result.get("filter_fallback_reason") == "metadata filter produced zero candidate chunks"

    def test_case_2_no_fallback_with_correct_filter(
        self, sample_chunks, mock_settings_fallback_enabled, caplog
    ):
        """Case 2: Query about day care with health filter should not trigger fallback."""
        query = "What is covered under day care procedures in this policy?"
        metadata_filter = {"insurance_type": "health"}
        
        with caplog.at_level(logging.INFO):
            results = hybrid_retrieve(
                sample_chunks,
                query,
                k=5,
                metadata_filter=metadata_filter,
                vector_store=None,
            )
        
        # Should return results without fallback
        assert len(results) > 0, "Should return results"
        
        # Check that no fallback was logged
        assert not any("Metadata filter returned zero chunks" in record.message 
                      for record in caplog.records)
        
        # Check that results do NOT have fallback metadata
        for result in results:
            assert result.get("fallback_used") is not True
            assert "original_metadata_filter" not in result

    def test_case_3_no_fallback_with_motor_query_and_motor_filter(
        self, sample_chunks, mock_settings_fallback_enabled, caplog
    ):
        """Case 3: Query about own damage with motor filter should not trigger fallback.
        
        Note: We create a separate list with actual motor chunks for this test.
        """
        # Create motor-specific chunks for this test
        motor_chunks = [
            Chunk(
                text="This motor insurance policy covers own damage to the vehicle.",
                source_id="motor_policy_1",
                source_path="/data/motor_policy_1.pdf",
                doc_type="policy_wording",
                insurance_type="motor",
                chunk_index=0,
                raw_metadata={"insurance_type": "motor"},
            ),
            Chunk(
                text="Third party liability coverage is included.",
                source_id="motor_policy_1",
                source_path="/data/motor_policy_1.pdf",
                doc_type="policy_wording",
                insurance_type="motor",
                chunk_index=1,
                raw_metadata={"insurance_type": "motor"},
            ),
        ]
        
        query = "What is covered under own damage?"
        metadata_filter = {"insurance_type": "motor"}
        
        with caplog.at_level(logging.INFO):
            results = hybrid_retrieve(
                motor_chunks,
                query,
                k=5,
                metadata_filter=metadata_filter,
                vector_store=None,
            )
        
        # Should return results without fallback
        assert len(results) > 0, "Should return results"
        
        # Check that no fallback was logged
        assert not any("Metadata filter returned zero chunks" in record.message 
                      for record in caplog.records)
        
        # Check that results do NOT have fallback metadata
        for result in results:
            assert result.get("fallback_used") is not True

    def test_fallback_disabled_returns_empty(
        self, sample_chunks, mock_settings_fallback_disabled, caplog
    ):
        """When fallback is disabled, should return empty results."""
        query = "What is covered under day care procedures in this policy?"
        metadata_filter = {"insurance_type": "motor"}
        
        with caplog.at_level(logging.WARNING):
            results = hybrid_retrieve(
                sample_chunks,
                query,
                k=5,
                metadata_filter=metadata_filter,
                vector_store=None,
            )
        
        # Should return empty results
        assert len(results) == 0, "Should return empty results when fallback disabled"
        
        # Check that fallback disabled was logged
        assert any("fallback disabled" in record.message for record in caplog.records)

    def test_logging_contains_required_fields(
        self, sample_chunks, mock_settings_fallback_enabled, caplog
    ):
        """Verify that logging contains all required fields."""
        query = "What is covered under day care procedures in this policy?"
        metadata_filter = {"insurance_type": "motor"}
        
        with caplog.at_level(logging.WARNING):
            hybrid_retrieve(
                sample_chunks,
                query,
                k=5,
                metadata_filter=metadata_filter,
                vector_store=None,
            )
        
        # Find the fallback log record
        fallback_record = None
        for record in caplog.records:
            if "Metadata filter returned zero chunks, applying fallback" in record.message:
                fallback_record = record
                break
        
        assert fallback_record is not None, "Should have fallback log record"
        
        # Check required fields - they are added directly to the LogRecord
        assert hasattr(fallback_record, "query")
        assert hasattr(fallback_record, "original_metadata_filter")
        assert hasattr(fallback_record, "chunks_before_filter")
        assert hasattr(fallback_record, "chunks_after_filter")
        assert hasattr(fallback_record, "fallback_used")
        assert hasattr(fallback_record, "reason")
        
        assert fallback_record.query == query
        assert fallback_record.original_metadata_filter == metadata_filter
        assert fallback_record.chunks_before_filter == 5
        assert fallback_record.chunks_after_filter == 0
        assert fallback_record.fallback_used is True
        assert "zero candidate chunks" in fallback_record.reason

    def test_no_filter_no_fallback(self, sample_chunks, mock_settings_fallback_enabled, caplog):
        """When no filter is provided, should not trigger fallback."""
        query = "What is covered under day care procedures?"
        
        with caplog.at_level(logging.INFO):
            results = hybrid_retrieve(
                sample_chunks,
                query,
                k=5,
                metadata_filter=None,
                vector_store=None,
            )
        
        # Should return results
        assert len(results) > 0
        
        # Check that no fallback was logged
        assert not any("Metadata filter returned zero chunks" in record.message 
                      for record in caplog.records)

    def test_empty_chunks_list(self, mock_settings_fallback_enabled):
        """When chunks list is empty, should return empty results."""
        query = "What is covered?"
        metadata_filter = {"insurance_type": "motor"}
        
        results = hybrid_retrieve(
            [],
            query,
            k=5,
            metadata_filter=metadata_filter,
            vector_store=None,
        )
        
        assert len(results) == 0

    def test_fallback_result_count_logged(
        self, sample_chunks, mock_settings_fallback_enabled, caplog
    ):
        """Verify that final result count is logged after fallback."""
        query = "What is covered under day care procedures in this policy?"
        metadata_filter = {"insurance_type": "motor"}
        
        with caplog.at_level(logging.WARNING):
            results = hybrid_retrieve(
                sample_chunks,
                query,
                k=5,
                metadata_filter=metadata_filter,
                vector_store=None,
            )
        
        # Find the "Filter fallback completed" log record
        completion_record = None
        for record in caplog.records:
            if "Filter fallback completed" in record.message:
                completion_record = record
                break
        
        assert completion_record is not None, "Should have completion log record"
        
        # Check final result count - fields are added directly to LogRecord
        assert hasattr(completion_record, "final_result_count")
        assert completion_record.final_result_count == len(results)
        assert completion_record.final_result_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])