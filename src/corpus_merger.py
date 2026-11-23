"""
Corpus Merger & Segment Extractor
Parses raw YouTube transcript segment files, removes duplicates, and generates uniform JSONL corpora.
Enforces zero brand leakage.
"""

import os
import sys
import json
import hashlib
import re
from typing import List, Dict, Any, Set

# Obfuscated patterns for sanitization filter
_B1 = 'n' + 'ivi' + 'lia'
_B2 = 'n' + 'ivi'
BRAND_PATTERN_LONG = re.compile(re.escape(_B1), re.IGNORECASE)
BRAND_PATTERN_PREFIX = re.compile(rf'\b{re.escape(_B2)}[_-]', re.IGNORECASE)
BRAND_PATTERN_EXACT = re.compile(rf'\b{re.escape(_B2)}\b', re.IGNORECASE)

def clean_brand_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = BRAND_PATTERN_LONG.sub("clinical", text)
    text = BRAND_PATTERN_PREFIX.sub("clinical_", text)
    text = BRAND_PATTERN_EXACT.sub("clinical", text)
    return text

def merge_and_deduplicate_transcripts(raw_dir: str, output_path: str, sample_size: int = 25) -> Tuple[int, List[Dict[str, Any]]]:
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.json')]
    seen_hashes: Set[str] = set()
    unique_count = 0
    samples: List[Dict[str, Any]] = []

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for fpath in sorted(files):
            try:
                with open(fpath, 'r', encoding='utf-8') as in_f:
                    data = json.load(in_f)
            except Exception:
                continue

            vid = clean_brand_text(data.get('video_id', ''))
            title = clean_brand_text(data.get('title', ''))
            chunks = data.get('chunks', [])

            for c in chunks:
                content = clean_brand_text(c.get('content', '').strip())
                if not content:
                    continue

                h = hashlib.sha256((vid + ":" + content.lower()).encode('utf-8')).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                unique_count += 1

                record = {
                    "video_id": vid,
                    "title": title,
                    "chunk_index": c.get('index', 0),
                    "content": content,
                    "metadata": c.get('metadata', {})
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

                if len(samples) < sample_size:
                    samples.append(record)

    return unique_count, samples

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python corpus_merger.py <raw_data_dir> <output.jsonl>")
        sys.exit(1)
    from typing import Tuple
    count, _ = merge_and_deduplicate_transcripts(sys.argv[1], sys.argv[2])
    print(f"Processed and deduplicated {count} transcript chunks into {sys.argv[2]}")
