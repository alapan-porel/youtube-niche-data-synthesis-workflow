"""
Corpus Merger & Segment Extractor
Parses raw YouTube transcript segment files, removes duplicates, and generates uniform JSONL corpora.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Set

def merge_and_deduplicate_transcripts(raw_dir: str, output_path: str) -> int:
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.json')]
    seen_hashes: Set[str] = set()
    unique_count = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for fpath in sorted(files):
            try:
                with open(fpath, 'r', encoding='utf-8') as in_f:
                    data = json.load(in_f)
            except Exception:
                continue

            vid = data.get('video_id', '')
            title = data.get('title', '')
            chunks = data.get('chunks', [])

            for c in chunks:
                content = c.get('content', '').strip()
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

    return unique_count
