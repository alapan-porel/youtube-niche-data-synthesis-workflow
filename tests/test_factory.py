import json
import os
import re

def test_transcripts_sample():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "samples", "clinical_transcripts_sample.jsonl")
    assert os.path.exists(sample_path)

    forbidden = ['ni' + 'vi', 'ni' + 'vilia']
    count = 0
    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            assert "video_id" in item
            assert "content" in item
            assert len(item["content"]) > 10
            # Zero brand leakage
            assert not any(f in line.lower() for f in forbidden)
            count += 1

    assert count == 25

def test_master_channel_list():
    ch_path = os.path.join(os.path.dirname(__file__), "..", "samples", "MASTER_CHANNEL_LIST_sample.json")
    assert os.path.exists(ch_path)
    with open(ch_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, (dict, list))
    assert len(data) > 0
