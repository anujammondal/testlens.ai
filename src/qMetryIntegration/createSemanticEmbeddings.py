"""
Create Normalized Semantic Representations of QMetry Test Cases

This script uses sentence-transformers/all-MiniLM-L6-v2 to:
1. Create a normalized text summary from test case steps
2. Generate embeddings from that summary for similarity comparison
"""

import json
import re
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not installed. Run:")
    print("   pip install sentence-transformers")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------- CONFIG ----------------
INPUT_FILE = "qmetry_testcases.json"
OUTPUT_FILE = "qmetry_testcases_embeddings.json"
EMBEDDINGS_NPY_FILE = "qmetry_embeddings.npy"
# all-mpnet-base-v2: Better accuracy, 768 dimensions (vs 384 for MiniLM)
# all-MiniLM-L6-v2: Faster, 384 dimensions, slightly less accurate
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ---------------- LOAD MODEL ----------------
print(f"🔄 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print(f"✅ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing special formatting.
    
    - Removes table formatting (||, |)
    - Removes extra whitespace
    - Removes special characters
    - Normalizes line breaks
    """
    if not text:
        return ""
    
    # Remove table formatting and data blocks
    text = re.sub(r'\|\|?\*?([^|*]+)\*?\|?\|?', r'\1', text)
    text = re.sub(r'\|+', ' ', text)
    
    # Remove markdown-like formatting
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    
    # Remove placeholder variables like <device>, <menuItems>, {name}, etc.
    text = re.sub(r'<\w+>', '', text)
    text = re.sub(r'\{\w+\}', '', text)
    text = re.sub(r'_\w+_', '', text)  # Remove _Region_ style placeholders
    
    # Remove data tables (lines with multiple colons or structured data)
    text = re.sub(r'\w+:\s*\w+\s*(?:,\s*\w+:\s*\w+)+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove technical notes like "Note: NA in Australia"
    text = re.sub(r'Note:\s*[^.]+\.?', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def extract_core_action(step_text: str) -> str:
    """
    Extract the core action/concept from a step description.
    Returns a clean, concise phrase describing what the step does.
    """
    # Remove BDD keywords
    for kw in ['GIVEN ', 'WHEN ', 'THEN ', 'AND ', 'BUT ', 'Given ', 'When ', 'Then ', 'And ', 'But ']:
        if step_text.startswith(kw):
            step_text = step_text[len(kw):]
            break
    
    # Simplify common phrases to their core meaning
    simplifications = [
        # Remove first-person pronouns and convert to action-focused
        (r'^I am an? ', ''),
        (r'^I have an? ', 'having '),
        (r'^I can ', 'can '),
        (r'^I should ', ''),
        (r'^I will ', ''),
        (r'^I am able to ', ''),
        (r'^I see ', 'shows '),
        (r'^I observe ', 'displays '),
        (r'^I verify ', ''),
        (r'^I validate ', ''),
        (r'^I navigate to ', 'navigating to '),
        (r'^I access ', 'accessing '),
        (r'^I select ', 'selecting '),
        (r'^I click ', 'clicking '),
        (r'^I tap ', 'tapping '),
        (r'^I enter ', 'entering '),
        (r'^I watch ', 'watching '),
        (r'^I play ', 'playing '),
        (r'^I launch ', 'launching '),
        (r'^I land ', 'landing '),
        (r'^I sign ', 'signing '),
        (r'^I ', ''),
        (r'^the user ', ''),
        (r'^user ', ''),
        # Remove filler words
        (r'correspondingly\s*', ''),
        (r'successfully\s*', ''),
        (r'properly\s*', ''),
        (r'correctly\s*', ''),
        (r'\s+for\s+corresponding\s*', ''),
        (r'\s+for\s+the\s+corresponding\s*', ''),
        (r'\s+for\s+$', ''),
        (r'\s+and\s+$', ''),
        # Clean up data markers
        (r'‹\w+›', ''),
        (r'with these mandatory data:.*', ''),
    ]
    
    for pattern, replacement in simplifications:
        step_text = re.sub(pattern, replacement, step_text, flags=re.IGNORECASE)
    
    # Truncate overly long steps (likely contain data tables)
    if len(step_text) > 80:
        # Try to find a natural break point
        for delimiter in ['. ', ', with ', ' with ', ' and ']:
            idx = step_text.find(delimiter)
            if 20 < idx < 70:
                step_text = step_text[:idx]
                break
        else:
            # Just truncate
            step_text = step_text[:70].rsplit(' ', 1)[0]
    
    # Final cleanup
    step_text = step_text.strip(' .,')
    
    return step_text


def create_normalized_summary(step_details: Dict[str, str], test_summary: str = "") -> str:
    """
    Create a lucid, goal-oriented summary describing the test case's target.
    
    Transforms step-by-step descriptions into a coherent statement of what
    the test case validates, focusing on the core functionality being tested.
    
    The output format is:
    "Test [context/precondition] by [action] to verify [expected outcome]"
    
    Args:
        step_details: Dictionary of step_N -> step description
        test_summary: Original test case summary (used to extract context)
    
    Returns:
        A lucid description of the test case's overall target/goal
    """
    # Get all step keys and sort them
    step_keys = sorted(step_details.keys(), key=lambda x: int(x.split('_')[1]))
    
    # Categorize steps by their BDD type
    preconditions = []  # GIVEN steps - setup/context
    actions = []        # WHEN steps - user actions
    validations = []    # THEN steps - expected outcomes
    
    for step_key in step_keys:
        step_text = clean_text(step_details.get(step_key, ""))
        if not step_text or len(step_text) < 5:
            continue
        
        upper_text = step_text.upper()
        
        # Categorize by BDD keyword
        if upper_text.startswith('GIVEN '):
            preconditions.append(extract_core_action(step_text))
        elif upper_text.startswith('WHEN '):
            actions.append(extract_core_action(step_text))
        elif upper_text.startswith('THEN '):
            validations.append(extract_core_action(step_text))
        elif upper_text.startswith('AND ') or upper_text.startswith('BUT '):
            # AND/BUT continues the previous category
            cleaned = extract_core_action(step_text)
            if validations:
                validations.append(cleaned)
            elif actions:
                actions.append(cleaned)
            elif preconditions:
                preconditions.append(cleaned)
        else:
            # No keyword - treat as action
            actions.append(extract_core_action(step_text))
    
    # Filter out empty/short items
    preconditions = [p for p in preconditions if len(p) > 5]
    actions = [a for a in actions if len(a) > 5]
    validations = [v for v in validations if len(v) > 5]
    
    # Build a lucid summary focusing on the test's goal
    # Format: "Validates [what] for [context] when [action]"
    
    summary_parts = []
    
    # Start with what's being validated (most important)
    if validations:
        # Use first validation as the main thing being tested
        main_validation = validations[0]
        summary_parts.append(f"Validates {main_validation}")
        
        # Add second validation if exists
        if len(validations) > 1:
            summary_parts.append(f"and {validations[1]}")
    
    # Add context (preconditions)
    if preconditions:
        context = preconditions[0]
        summary_parts.append(f"for {context}")
    
    # Add the action/trigger
    if actions:
        action = actions[0]
        summary_parts.append(f"when {action}")
    
    # If no validations, build differently
    if not validations:
        summary_parts = []
        if actions:
            summary_parts.append(f"Tests {actions[0]}")
            if len(actions) > 1:
                summary_parts.append(f"and {actions[1]}")
        if preconditions:
            summary_parts.append(f"for {preconditions[0]}")
    
    # Combine into final summary
    normalized = " ".join(summary_parts)
    
    # Final cleanup - fix grammar issues
    cleanup_patterns = [
        (r'\s+', ' '),                          # Multiple spaces
        (r'\.+', '.'),                          # Multiple periods
        (r',\s*,', ','),                        # Multiple commas
        (r'\s+([,.])', r'\1'),                  # Space before punctuation
        # Fix validation phrases
        (r'Validates for I see', 'Validates that'),
        (r'Validates see ', 'Validates that '),
        (r'Validates displays', 'Validates that it displays'),
        (r'Validates shows', 'Validates that it shows'),
        (r'Validates land', 'Validates landing'),
        (r'Validates signing', 'Validates sign-out'),
        (r'Validates be taken', 'Validates being taken'),
        (r'Validates starts', 'Validates starting'),
        (r'Validates It should', 'Validates that content'),
        # Fix prepositions and flow
        (r'when can see', 'when seeing'),
        (r'when shows', 'when shown'),
        (r'when choose', 'when choosing'),
        (r'when download', 'when downloading'),
        (r'when selecting', 'upon selecting'),
        (r'when navigating', 'upon navigating'),
        (r'when accessing', 'upon accessing'),
        (r'when watching', 'while watching'),
        (r'when launching', 'after launching'),
        (r'for launching', 'after launching'),
        (r'for have the', 'for users who have the'),
        (r'for authenticated', 'for an authenticated'),
        (r'for unauthenticated', 'for an unauthenticated'),
        (r'for lapsed', 'for a lapsed'),
        (r'for MVPD', 'for an MVPD'),
        # Clean up incomplete phrases
        (r'navigate to and ', 'navigation and '),
        (r'land on for ', 'landing on home screen for '),
        (r'sign-out out', 'sign-out'),
        (r' and when I select', ', selecting'),
        (r' and After I', ', after'),
        (r' and selecting ', ' and '),
        (r' I see modals asking if the$', ''),
        (r' Note - .*$', ''),
        # Remove duplicates
        (r' and and ', ' and '),
        (r' or or ', ' or '),
        # Remove trailing incomplete phrases
        (r'\s+for\s*$', ''),
        (r'\s+and\s*$', ''),
        (r'\s+when\s*$', ''),
        (r'\s+upon\s*$', ''),
        (r'\s+after\s*$', ''),
        (r'\s+a\s*$', ''),
        (r'\s+the\s*$', ''),
    ]
    
    for pattern, replacement in cleanup_patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE if pattern.startswith(r'Validates') else 0)
    
    normalized = normalized.strip(' .,')
    
    # Ensure it ends properly
    if normalized:
        normalized += '.'
        # Capitalize first letter
        normalized = normalized[0].upper() + normalized[1:]
    
    return normalized


def create_embedding(text: str) -> List[float]:
    """Create normalized embedding for text."""
    if not text:
        return [0.0] * model.get_sentence_embedding_dimension()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def process_test_cases(input_file: str) -> Dict[str, Any]:
    """
    Process all test cases and create semantic embeddings.
    
    Args:
        input_file: Path to the QMetry test cases JSON file
    
    Returns:
        Dictionary containing test cases with normalized summaries and embeddings
    """
    print(f"\n📂 Loading test cases from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    testcases = data.get("testcases", [])
    total = len(testcases)
    print(f"📊 Found {total} test cases to process")
    
    processed_testcases = []
    all_embeddings = []
    
    for i, tc in enumerate(testcases, 1):
        tc_key = tc.get("key", "Unknown")
        summary = tc.get("summary", "")
        step_details = tc.get("stepDetails", {})
        expected_results = tc.get("expectedResults", {})
        
        # Create normalized summary - a lucid description of the test's target
        normalized_summary = create_normalized_summary(step_details, summary)
        
        # Extract key action steps (clean them for embedding)
        key_steps = []
        for step_key in sorted(step_details.keys()):
            step_text = step_details.get(step_key, "")
            if step_text:
                # Clean and extract core content from step
                cleaned_step = clean_text(step_text)
                if cleaned_step and len(cleaned_step) > 10:
                    key_steps.append(cleaned_step)
        
        # Create embedding from combined text: summary + normalized + key steps
        # This provides rich semantic context for better similarity matching
        steps_text = " ".join(key_steps[:5])  # Use first 5 cleaned steps
        combined_text = f"{summary}. {normalized_summary}. Steps: {steps_text}"
        embedding = create_embedding(combined_text)
        all_embeddings.append(embedding)
        
        # Create processed test case
        processed_tc = {
            "id": tc.get("id"),
            "key": tc_key,
            "folder": tc.get("folder", {}),
            "summary": summary,
            "priority": tc.get("priority", {}).get("name", ""),
            "stepsCount": tc.get("stepsCount", 0),
            "stepDetails": step_details,
            "expectedResults": expected_results,
            "normalized_summary": normalized_summary,
            "embedding": embedding
        }
        
        processed_testcases.append(processed_tc)
        
        if i % 10 == 0 or i == total:
            print(f"   ⚙️ Processed {i}/{total} test cases")
    
    return {
        "model": MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "total_testcases": total,
        "projectId": data.get("projectId", ""),
        "parentFolder": data.get("parentFolder", {}),
        "testcases": processed_testcases
    }, np.array(all_embeddings)


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two normalized embeddings.
    Since embeddings are already normalized, dot product = cosine similarity.
    """
    return float(np.dot(embedding1, embedding2))


def find_similar_testcases(
    query_embedding: List[float],
    all_embeddings: np.ndarray,
    testcases: List[Dict],
    top_k: int = 5,
    exclude_self: bool = False
) -> List[Dict]:
    """
    Find the most similar test cases to a query embedding.
    
    Args:
        query_embedding: The query embedding vector
        all_embeddings: Numpy array of all test case embeddings
        testcases: List of test case dictionaries
        top_k: Number of similar test cases to return
        exclude_self: If True, exclude exact matches (similarity = 1.0)
    
    Returns:
        List of similar test cases with similarity scores
    """
    query = np.array(query_embedding)
    
    # Compute similarities (dot product since normalized)
    similarities = np.dot(all_embeddings, query)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        
        # Skip exact matches if requested
        if exclude_self and sim > 0.9999:
            continue
            
        tc = testcases[idx]
        results.append({
            "key": tc["key"],
            "summary": tc["summary"],
            "similarity": sim,
            "stepsCount": tc["stepsCount"],
            "normalized_summary": tc.get("normalized_summary", "")[:200] + "..."
        })
        
        if len(results) >= top_k:
            break
    
    return results


def save_outputs(result: Dict, embeddings: np.ndarray, output_json: str, output_npy: str):
    """Save the processed results to files."""
    
    # Save JSON
    print(f"\n💾 Saving results to: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save embeddings as numpy array for efficient loading
    print(f"💾 Saving embeddings to: {output_npy}")
    np.save(output_npy, embeddings)
    
    print(f"✅ Saved {result['total_testcases']} test cases with normalized summaries and embeddings")


def demo_similarity_search(result: Dict, embeddings: np.ndarray):
    """Demonstrate similarity search with sample queries."""
    
    testcases = result["testcases"]
    
    print("\n" + "="*70)
    print("🔍 DEMO: Similarity Search")
    print("="*70)
    
    # Show first test case's normalized summary
    query_tc = testcases[0]
    print(f"\n📋 Sample Test Case: {query_tc['key']}")
    print(f"   Original Summary: {query_tc['summary'][:80]}...")
    print(f"\n   Normalized Summary:")
    print(f"   {query_tc['normalized_summary'][:300]}...")
    
    # Find similar test cases
    query_embedding = query_tc["embedding"]
    similar = find_similar_testcases(
        query_embedding, embeddings, testcases, 
        top_k=5, exclude_self=True
    )
    
    print(f"\n🔗 Top 5 Similar Test Cases:")
    for i, s in enumerate(similar, 1):
        print(f"\n   {i}. {s['key']} (similarity: {s['similarity']:.4f})")
        print(f"      Summary: {s['summary'][:70]}...")
    
    # Demo: Search with custom text
    print("\n" + "-"*70)
    print("🔍 DEMO: Search with Custom Queries")
    print("-"*70)
    
    custom_queries = [
        "user login authentication verification",
        "video playback streaming content",
        "subscription payment purchase",
        "app navigation deep link"
    ]
    
    for query in custom_queries:
        print(f"\n📝 Query: '{query}'")
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()
        similar = find_similar_testcases(query_embedding, embeddings, testcases, top_k=3)
        
        for i, s in enumerate(similar, 1):
            print(f"   {i}. {s['key']} (sim: {s['similarity']:.3f}) - {s['summary'][:50]}...")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create normalized semantic embeddings for QMetry test cases"
    )
    parser.add_argument(
        "--input", "-i",
        default=INPUT_FILE,
        help=f"Input JSON file (default: {INPUT_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_FILE,
        help=f"Output JSON file (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--embeddings", "-e",
        default=EMBEDDINGS_NPY_FILE,
        help=f"Output numpy embeddings file (default: {EMBEDDINGS_NPY_FILE})"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run similarity search demo after processing"
    )
    
    args = parser.parse_args()
    
    # Process test cases
    result, embeddings = process_test_cases(args.input)
    
    # Save outputs
    save_outputs(result, embeddings, args.output, args.embeddings)
    
    # Run demo if requested
    if args.demo:
        demo_similarity_search(result, embeddings)
    
    print("\n✅ Done!")
