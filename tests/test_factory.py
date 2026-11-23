import json
import os
import re
import unittest
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestYouTubeFactory(unittest.TestCase):
    def test_transcripts_sample(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "samples", "clinical_transcripts_sample.jsonl")
        self.assertTrue(os.path.exists(sample_path), f"Sample file not found at {sample_path}")

        forbidden_pattern = re.compile(r'\b(ni' + 'vi|ni' + 'vilia)\b', re.IGNORECASE)
        count = 0
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                self.assertIn("video_id", item)
                self.assertIn("content", item)
                self.assertGreater(len(item["content"]), 10)
                # Zero brand leakage
                self.assertFalse(forbidden_pattern.search(line), f"Brand leak detected: {line}")
                count += 1

        self.assertEqual(count, 25)

    def test_master_channel_list(self):
        ch_path = os.path.join(os.path.dirname(__file__), "..", "samples", "MASTER_CHANNEL_LIST_sample.json")
        self.assertTrue(os.path.exists(ch_path), f"Channel list not found at {ch_path}")
        with open(ch_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, (dict, list))
        self.assertGreater(len(data), 0)

if __name__ == "__main__":
    unittest.main()
