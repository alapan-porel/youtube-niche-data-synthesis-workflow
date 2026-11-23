"""
clinical ENGINE 1: THE SCOUT V9.2 (Final Polish)
============================================
1. FINDS videos using Title + Description context.
2. FILTERS in mini-batches (10 at a time).
3. FIXED: 'Left Padding' enabled for 100% AI accuracy.
4. SAVES full data to 'clinical_targets.jsonl'.
"""

import os
import json
import warnings
import socket
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(60)

# ================= CONFIGURATION =================
# Loads API keys from environment variable YOUTUBE_API_KEYS (comma-separated)
_env_keys = os.getenv("YOUTUBE_API_KEYS", "")
API_KEY_POOL = [k.strip() for k in _env_keys.split(",") if k.strip()] or [
    "YOUR_YOUTUBE_API_KEY_HERE"
]

TARGET_FILE = "clinical_targets.jsonl"
LOG_FILE = "scanned_channels.log"
BATCH_SIZE = 10 

# ================= KEY MANAGER =================
class KeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.current_index = 0
    def get_service(self):
        if self.current_index >= len(self.keys): return None
        return build("youtube", "v3", developerKey=self.keys[self.current_index], static_discovery=False)
    def rotate(self):
        print(f"\n⚠️ Quota Hit. Switching Keys...")
        self.current_index += 1
        return self.get_service()

key_mgr = KeyManager(API_KEY_POOL)
youtube = key_mgr.get_service()

# ================= GATEKEEPER (AI FILTER) =================
print("🧠 Loading Gatekeeper (Qwen 2.5)...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# ✅ FIX: Explicitly set padding to LEFT for decoder-only models
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype="auto").eval()

def batch_filter(videos):
    """Filters videos in safe batches to prevent OOM errors."""
    kept = []
    if not videos: return []
    
    system_msg = """You are an Expert Data Curator for an AI Mental Health Platform.
    YOUR GOAL: Select videos that contain human conversations, personal stories, advice, therapy, or emotional expression.
    REJECT (Output 'NO'): Gaming, Tech, Politics, Music, Memes.
    KEEP (Output 'YES'): Personal Stories, Psychology, Vlogs, Podcasts.
    Output ONLY 'YES' or 'NO'."""

    for i in range(0, len(videos), BATCH_SIZE):
        batch = videos[i : i + BATCH_SIZE]
        prompts = []
        
        for v in batch:
            desc_snippet = v['description'][:300].replace("\n", " ") 
            user_msg = f"Channel: {v['channel']}\nTitle: {v['title']}\nDescription: {desc_snippet}\nKeep?"
            text = tokenizer.apply_chat_template([{"role": "system", "content": system_msg}, 
                                                  {"role": "user", "content": user_msg}], 
                                                  tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, temperature=0.1)
        
        responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for j, resp in enumerate(responses):
            if "NO" not in resp.strip().upper():
                kept.append(batch[j])
            
    return kept

# ================= SCANNER LOGIC =================
def get_channel_videos(cid, c_meta):
    global youtube
    videos = []
    try:
        res = youtube.channels().list(id=cid, part="contentDetails").execute()
        if not res.get("items"): return []
        uploads_id = res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        next_page = None
        count = 0
        while count < 500: 
            try:
                pl = youtube.playlistItems().list(playlistId=uploads_id, part="snippet", maxResults=50, pageToken=next_page).execute()
                for item in pl["items"]:
                    videos.append({
                        "id": item["snippet"]["resourceId"]["videoId"],
                        "title": item["snippet"]["title"],
                        "description": item["snippet"].get("description", ""), 
                        "channel": c_meta.get("name", "Unknown"),
                        "meta": c_meta
                    })
                    count += 1
                next_page = pl.get("nextPageToken")
                if not next_page: break
            except HttpError as e:
                if "quotaExceeded" in str(e):
                    youtube = key_mgr.rotate()
                    if not youtube: return videos
                    continue
                else: break
    except: pass
    return videos

# ================= MAIN =================
def main():
    try:
        with open("MASTER_CHANNEL_LIST.json", "r") as f:
            master_list = json.load(f)
    except FileNotFoundError:
        print("❌ ERROR: MASTER_CHANNEL_LIST.json missing.")
        return
    
    scanned = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            scanned = set(line.strip() for line in f)
            
    print(f"📋 Master List: {len(master_list)} channels.")
    print(f"⏭️ Skipping {len(scanned)} already scanned channels.")
    
    channels_to_scan = [cid for cid in master_list.keys() if cid not in scanned]
    pbar = tqdm(channels_to_scan)
    
    for cid in pbar:
        if not youtube: break
        
        # 1. Fetch
        raw_videos = get_channel_videos(cid, master_list[cid])
        if not raw_videos: continue
        
        # 2. Filter 
        good_videos = batch_filter(raw_videos)
        
        pbar.set_description(f"Hits: {len(good_videos)}")
        
        # 3. SAVE IMMEDIATELY 
        if good_videos:
            with open(TARGET_FILE, "a", encoding="utf-8") as f:
                for v in good_videos:
                    f.write(json.dumps(v) + "\n") 
        
        # 4. Mark Done
        with open(LOG_FILE, "a") as log:
            log.write(f"{cid}\n")

if __name__ == "__main__":
    main()