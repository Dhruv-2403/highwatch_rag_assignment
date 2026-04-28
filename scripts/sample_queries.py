#!/usr/bin/env python3
"""
Demo script: Run sample queries without Google Drive credentials.
Creates synthetic policy documents and tests the RAG pipeline.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Sample documents
SAMPLE_DOCS = {
    "refund_policy.txt": """
REFUND POLICY

1. ELIGIBILITY
Customers are eligible for a full refund within 30 days of purchase if the product is unused and in its original packaging.

2. AFTER 30 DAYS
After 30 days, refunds are only available for defective products. Customers must provide proof of defect.

3. PROCESS
To request a refund, customers should contact support@acme.com with their order number and reason for return.

4. TIMELINE
Refunds are processed within 5-7 business days after approval.
""",
    "compliance_policy.txt": """
COMPLIANCE POLICY

1. ANTI-BRIBERY
Gifts to clients or partners must not exceed $50 in value. Any exceptions require written approval from the Legal department.

2. DATA PRIVACY
All customer data must be handled in compliance with GDPR. No third-party sharing without explicit customer consent.

3. CONFLICT OF INTEREST
Employees must disclose any potential conflicts of interest within 30 days of becoming aware of them.

4. INCIDENT REPORTING
Suspected compliance violations should be reported to compliance@acme.com or through the anonymous whistleblower hotline at 1-800-ACM-SAFE.
""",
    "onboarding_sop.txt": """
EMPLOYEE ONBOARDING SOP

DAY 1:
- HR sends welcome email with system credentials by 8 AM
- IT provisions laptop and access to Slack, Jira, and Confluence
- Manager schedules introductory 1:1 meeting

WEEK 1:
- Complete mandatory compliance training
- Meet with team members
- Review company handbook and policies

WEEK 2:
- Attend department orientation
- Set up development environment
- Begin assigned projects

ONGOING:
- 30-day check-in with manager
- 90-day performance review
""",
}


def setup_demo():
    """Create sample documents and initialize the RAG system."""
    from processing.pipeline import process_drive_file, Chunk
    from embedding.encoder import encode
    from search.store import get_store
    from connectors.gdrive import DriveFile

    logger.info("Setting up demo with sample documents...")

    store = get_store()
    store._ensure_loaded()

    # Clear existing data
    store._index = None
    store._metadata = []
    store._loaded = False

    all_chunks = []
    all_embeddings = []

    for file_name, content in SAMPLE_DOCS.items():
        # Create a synthetic DriveFile
        drive_file = DriveFile(
            file_id=file_name.replace(".txt", ""),
            file_name=file_name,
            mime_type="text/plain",
            content=content.encode("utf-8"),
            web_view_link=f"https://drive.google.com/file/d/{file_name}",
            modified_time="2026-04-25T10:00:00Z",
        )

        # Process the file
        chunks = process_drive_file(drive_file)
        all_chunks.extend(chunks)

        logger.info(f"  ✓ {file_name}: {len(chunks)} chunks")

    # Encode all chunks
    logger.info("Encoding chunks...")
    texts = [c.text for c in all_chunks]
    embeddings = encode(texts, show_progress=False)

    # Store in FAISS
    logger.info("Storing in vector database...")
    store.upsert_chunks(all_chunks, embeddings.astype(np.float32))

    stats = store.stats()
    logger.info(
        f"✓ Demo ready: {stats['total_chunks']} chunks from {stats['unique_documents']} documents"
    )


def run_queries():
    """Run sample queries against the demo data."""
    from embedding.encoder import encode_single
    from search.store import get_store

    store = get_store()

    queries = [
        "What is our refund policy?",
        "What are our company policies on compliance?",
        "What happens on an employee's first day?",
        "Who do I report a compliance violation to?",
    ]

    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SAMPLE QUERIES (RAG Retrieval Demo)")
    logger.info("=" * 80 + "\n")

    for query in queries:
        logger.info(f"Q: {query}")

        # Encode query
        query_vec = encode_single(query)

        # Search
        results = store.search(query_embedding=query_vec, top_k=3)

        if not results:
            logger.info("A: No relevant documents found.\n")
            continue

        # Show retrieved context (RAG retrieval working)
        try:
            # Concatenate top results as context
            context_text = "\n\n".join([r.text for r in results])
            answer = f"Based on the documents:\n{context_text[:400]}..."
            logger.info(f"A: {answer}\n")

            # Show sources with relevance scores
            sources = list({r.file_name for r in results})
            logger.info(f"Sources: {sources}")
            for r in results:
                logger.info(f"  - {r.file_name}: relevance={r.score:.4f}")
        except Exception as e:
            logger.error(f"Error processing query: {e}\n")

        logger.info("-" * 80 + "\n")


if __name__ == "__main__":
    try:
        setup_demo()
        run_queries()
        logger.info("✓ Demo completed successfully!")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        exit(1)
