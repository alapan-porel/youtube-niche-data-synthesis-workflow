"""
clinical YouTube Training Data Generator v4.2
==========================================
Generates LLaMA 3.1 fine-tuning conversations from verified therapist YouTube channels.

Pipeline:
1. Download audio via yt-dlp
2. Transcribe via Gemini 1.5 Flash
3. Filter therapeutic content
4. Classify risk level
5. Synthesize realistic conversations via Claude
6. Quality check & route to training files
7. Track costs & generate reports

Author: clinical Team
Last Updated: 2025-01-08
"""

import os
import json
import time
import re
import hashlib
import random
import textwrap
from tqdm import tqdm
from datetime import datetime

import yt_dlp
import ollama
import anthropic
import whisper 
import warnings
warnings.filterwarnings("ignore")
from googleapiclient.discovery import build

# ============================================================
# ======================== CONFIGURATION ======================
# ============================================================

# API Keys (Set these as environment variables or replace directly)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")

print("⏳ Loading local Whisper model (this takes a moment)...")
# "small" is a good balance of speed/quality. Use "base" if you have a slow CPU.
local_whisper_model = whisper.load_model("base")

# Verified Therapist Channels (Manually verified)
CHANNEL_IDS = [
    "UCftwRNsjfRo08xYE31tkiyw",  # ⭐ MANUAL | WIRED (12,600,000 subs)
    # Add more verified channels here
]

# Channel Metadata (for training context)
CHANNEL_METADATA = {
    "UCftwRNsjfRo08xYE31tkiyw": {
        "name": "WIRED",
        "credentials": "Science & Tech Journalists",
        "quality_tier": "B",
        "specialization": ["Science", "Health", "Technology"]
    },
}

# Content Filtering
EXCLUDED_KEYWORDS = [
    "sponsor", "sponsorship", "ad", "advertisement", "merch", "merchandise",
    "vlog", "daily vlog", "q&a", "q and a", "reacting to", "reaction",
    "try", "trying", "challenge", "tour", "update", "announcement",
    "behind the scenes", "bts", "unboxing", "haul", "shopping"
]

REQUIRED_KEYWORDS = [
    # Core mental health
    "mental health", "mental wellbeing", "mental well-being", "emotional health",
    "psychological health", "therapy", "therapist", "counseling", "counselling",
    "psychology", "psychiatry", "psychological",

    # Disorders & conditions
    "depression", "anxiety", "panic", "panic attack", "stress", "burnout",
    "trauma", "ptsd", "complex trauma", "cptsd",
    "adhd", "add", "bipolar", "ocd", "phobia",
    "eating disorder", "disordered eating",
    "addiction", "substance abuse", "dependency",

    # Emotional states & struggles
    "emotional", "emotions", "overwhelmed", "exhausted", "numb",
    "lonely", "isolation", "grief", "loss", "sadness",
    "anger", "rage", "guilt", "shame", "fear", "hopelessness",
    "self-doubt", "insecurity", "low self-esteem",
    "emptiness", "despair", "emotional pain",

    # Coping & regulation
    "coping", "coping skills", "emotional regulation", "self-regulation",
    "grounding", "grounding techniques",
    "stress management", "anger management",
    "boundaries", "healthy boundaries",
    "self-soothing", "calming", "nervous system",

    # Therapy & techniques
    "cbt", "cognitive behavioral therapy",
    "dbt", "dialectical behavior therapy",
    "act", "acceptance and commitment therapy",
    "psychoeducation", "trauma-informed",
    "inner child", "attachment", "attachment style",
    "somatic", "somatic healing",

    # Self-care & healing
    "self-care", "healing", "recovery", "rest",
    "mindfulness", "meditation", "breathing",
    "self-compassion", "compassion",
    "resilience", "emotional resilience",

    # Identity & self-concept
    "self-awareness", "self-reflection",
    "self-worth", "self-respect",
    "identity", "sense of self",
    "people pleasing", "perfectionism",

    # Relationships & social
    "relationships", "relationship issues",
    "communication", "emotional safety",
    "trust", "attachment wounds",
    "conflict", "family dynamics",
    "toxic relationships", "codependency",

    # Life challenges & growth
    "life stress", "life transitions",
    "career stress", "work stress",
    "motivation", "burnout recovery",
    "personal growth", "inner growth",
    "meaning", "purpose",
    "values", "goals",

    # Crisis & distress signals (important to include)
    "emotional distress", "crisis",
    "breakdown", "mental breakdown",
    "feeling stuck", "feeling lost",
    "overthinking", "rumination",

    # General but high-signal phrases
    "how to cope", "why do i feel",
    "what am i feeling",
    "mental struggles", "emotional struggles",
    "healing journey", "therapy session"
]

# Canonical Interventions (for metadata consistency)
CANONICAL_INTERVENTIONS = [
    "Validation",
    "Metacognitive Observation",
    "Cognitive Restructuring",
    "Somatic Regulation",
    "Behavioral Activation",
    "Motivational Interviewing",
    "Problem Solving",
    "Psychoeducation",
    "Grounding Techniques",
    "Safety Planning"
]

# Processing Configuration
CHUNK_SIZE = 4500  # Characters per chunk for processing
DOWNLOAD_DIR = "temp_audio"
RATE_LIMIT_SECONDS = 30.0  # Delay between API calls

# Output Files
LEDGER_FILE = "processing_ledger.jsonl"
TRAINING_DATA = "llama_training_data.jsonl"
REVIEW_QUEUE = "manual_review_queue.jsonl"
COST_REPORT = "cost_report.json"

# ============================================================
# ====================== INITIALIZE CLIENTS ===================
# ============================================================

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ============================================================
# ====================== COST TRACKING ========================
# ============================================================

class CostTracker:
    """Track API costs and usage statistics"""
    
    def __init__(self):
        self.costs = {
            "gemini_audio": 0.0,
            "gemini_text": 0.0,
            "claude_synthesis": 0.0,
            "total_videos": 0,
            "total_chunks": 0,
            "total_conversations": 0,
            "filtered_videos": 0,
            "failed_transcriptions": 0
        }
        self.start_time = datetime.utcnow()
    
    def log_gemini_audio(self, duration_minutes):
        """Log Gemini audio transcription cost"""
        # Gemini 1.5 Flash: ~$0.075 per minute of audio
        cost = duration_minutes * 0.075
        self.costs["gemini_audio"] += cost
        return cost
    
    def log_gemini_text(self, input_chars, output_chars):
        """Log Gemini text generation cost"""
        # Rough estimate: $0.0001 per 1K chars
        cost = (input_chars + output_chars) / 1000 * 0.0001
        self.costs["gemini_text"] += cost
        return cost
    
    def log_claude(self, input_tokens, output_tokens, model="haiku"):
        """Log Claude synthesis cost"""
        if model == "sonnet":
            # Claude Sonnet: $3.00 / 1M input, $15.00 / 1M output
            cost = (input_tokens / 1_000_000 * 3.00 + 
                    output_tokens / 1_000_000 * 15.00)
        else:
            # Claude Haiku: $0.25 / 1M input, $1.25 / 1M output
            cost = (input_tokens / 1_000_000 * 0.25 + 
                    output_tokens / 1_000_000 * 1.25)
        self.costs["claude_synthesis"] += cost
        return cost
    
    def report(self):
        """Print cost and usage report"""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        
        print("\n" + "="*60)
        print("💰 COST & USAGE REPORT")
        print("="*60)
        print(f"\n⏱️  Runtime: {elapsed/60:.1f} minutes")
        print(f"\n📹 Video Processing:")
        print(f"  • Total Videos: {self.costs['total_videos']}")
        print(f"  • Filtered Out: {self.costs['filtered_videos']}")
        print(f"  • Failed Transcriptions: {self.costs['failed_transcriptions']}")
        print(f"  • Total Chunks: {self.costs['total_chunks']}")
        print(f"\n💬 Output:")
        print(f"  • Total Conversations: {self.costs['total_conversations']}")
        print(f"  • Conversations per Video: {self.costs['total_conversations'] / max(1, self.costs['total_videos']):.1f}")
        print(f"\n💵 Costs:")
        print(f"  • Gemini Audio: ${self.costs['gemini_audio']:.2f}")
        print(f"  • Gemini Text: ${self.costs['gemini_text']:.2f}")
        print(f"  • Claude Synthesis: ${self.costs['claude_synthesis']:.2f}")
        
        total_cost = sum([v for k, v in self.costs.items() if isinstance(v, float)])
        print(f"  • TOTAL COST: ${total_cost:.2f}")
        
        if self.costs['total_conversations'] > 0:
            print(f"  • Cost per Conversation: ${total_cost / self.costs['total_conversations']:.4f}")
        
        print("\n" + "="*60)
        
        # Save detailed report
        with open(COST_REPORT, "w") as f:
            json.dump({
                "costs": self.costs,
                "runtime_minutes": elapsed / 60,
                "timestamp": datetime.utcnow().isoformat()
            }, f, indent=2)

cost_tracker = CostTracker()

# ============================================================
# ====================== UTILITIES ============================
# ============================================================

def clean_json(text):
    """Remove markdown code blocks from JSON responses"""
    if not text:
        return None
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()

def compute_hash(video_id, chunk_idx, variation_idx):
    """Generate unique hash for deduplication"""
    raw = f"{video_id}|{chunk_idx}|{variation_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()

# ============================================================
# ====================== LEDGER (DEDUPLICATION) ===============
# ============================================================

SEEN_HASHES = set()

if os.path.exists(LEDGER_FILE):
    with open(LEDGER_FILE, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                SEEN_HASHES.add(entry["hash"])
            except:
                pass

def mark_seen(hash_id, meta):
    """Mark a conversation as processed"""
    SEEN_HASHES.add(hash_id)
    with open(LEDGER_FILE, "a") as f:
        f.write(json.dumps({
            "hash": hash_id,
            "meta": meta,
            "timestamp": datetime.utcnow().isoformat()
        }) + "\n")

# ============================================================
# ========= STAGE 1: AUDIO DOWNLOAD & TRANSCRIPTION ===========
# ============================================================

def download_and_transcribe(video_id):
    """
    Download audio locally and transcribe using OpenAI Whisper (Verbose Mode)
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = os.path.join(DOWNLOAD_DIR, video_id)
    audio_path = f"{base}.mp3"
    
    print(f"\n🎧 [Starting] {video_id}...") # Log start
    
    # 1. DOWNLOAD: Turn OFF quiet mode so you see the bar
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": base,
        "quiet": False,   # <--- CHANGED: Show download progress
        "no_warnings": True,
        "cookiefile": "cookies.txt", 
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "ignoreerrors": True,
    }
    
    try:
        # Download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration_minutes = info.get("duration", 0) / 60.0
        
        if not os.path.exists(audio_path):
            return {"success": False, "error": "Download failed"}
        
        # 2. TRANSCRIBE: Turn ON verbose mode to see text
        print(f"📝 [Transcribing] {video_id} (This uses your CPU)...")
        try:
            # verbose=True prints the text as it recognizes it!
            result = local_whisper_model.transcribe(audio_path, verbose=True) 
            transcript = result["text"].strip()
            
            cost_tracker.costs["gemini_audio"] += 0.0 
            
            return {
                "success": True,
                "text": transcript,
                "duration_minutes": duration_minutes
            }

        except Exception as e:
            raise Exception(f"Local Whisper Error: {e}")

    except Exception as e:
        print(f"❌ Transcription error for {video_id}: {str(e)}")
        cost_tracker.costs["failed_transcriptions"] += 1
        return {"success": False, "error": str(e)}
    
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass

# ============================================================
# ========= STAGE 2: CONTENT FILTERING ========================
# ============================================================

def is_therapeutic_content(title, description=""):
    """
    Filter out non-therapeutic videos (Keyword check)
    """
    title_lower = title.lower()
    desc_lower = description.lower() if description else ""
    
    # Exclude non-therapeutic content
    for keyword in EXCLUDED_KEYWORDS:
        if keyword in title_lower:
            return False
    
    # Require at least one therapeutic keyword
    for keyword in REQUIRED_KEYWORDS:
        if keyword in title_lower or keyword in desc_lower:
            return True
    
    return False

def gate_relevance_and_style(title, text_chunk):
    """
    Use LOCAL QWEN (Ollama) to verify relevance and detect delivery style.
    Features: Checks relevance, detects 1 of 5 styles, gives confidence score.
    """
    prompt = f"""
    Analyze this video transcript segment. Return ONLY valid JSON.
    
    TASK 1: Relevance
    - Is this therapeutic mental health content? (true/false)
    - (Ignore gaming, marketing, or casual vlogs)

    TASK 2: Delivery Style (Pick ONE)
    - "Warm & Maternal" (Comforting, gentle, validating)
    - "Direct & Coach-like" (Action-oriented, firm, tough love)
    - "Quiet & Contemplative" (Slow, philosophical, deep)
    - "Playful & Metaphorical" (Uses humor, stories, analogies)
    - "Scientific & Grounded" (Clinical, neurological explanations)

    VIDEO TITLE: {title}
    CONTENT: {text_chunk[:1000]} 

    OUTPUT FORMAT (JSON ONLY):
    {{
      "relevant": true,
      "delivery_style": "Scientific & Grounded",
      "confidence": 0.95
    }}
    """
    
    try:
        # CALL LOCAL MODEL
        response = ollama.chat(model='qwen2.5:1.5b', messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        # Clean response (sometimes models add ```json ... ```)
        content = response['message']['content'].replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        # Fallback for style if model forgets it
        if "delivery_style" not in result:
            result["delivery_style"] = "Warm & Maternal"
            
        return result
    
    except Exception as e:
        print(f"⚠️ Local Gate error: {e}")
        # Fail safe: Assume relevant so we don't lose data, default to Warm style
        return {"relevant": True, "confidence": 0.5, "delivery_style": "Warm & Maternal"}

# ============================================================
# ========= STAGE 3: RISK CLASSIFICATION ======================
# ============================================================

def classify_risk_intent(text_chunk):
    """
    Use LOCAL QWEN (Ollama) to classify risk level.
    """
    prompt = f"""
    Classify the mental health risk. Return ONLY valid JSON.
    
    RISK LEVELS:
    - "NONE": Educational, general advice, normal conversation.
    - "DISTRESS": Emotional pain, crying, overwhelmed, sad.
    - "CRISIS": Self-harm, suicide references, immediate danger.

    GUIDELINES:
    - If RISK is "NONE" -> recommended_mode is "THERAPY"
    - If RISK is "DISTRESS" -> recommended_mode is "SUPPORT"
    - If RISK is "CRISIS" -> recommended_mode is "REFERRAL" (Must provide 988)

    CONTENT: {text_chunk[:1000]}

    OUTPUT FORMAT (JSON ONLY):
    {{
      "risk_intent": "NONE",
      "confidence": 0.9,
      "recommended_mode": "THERAPY"
    }}
    """
    
    try:
        # CALL LOCAL MODEL
        response = ollama.chat(model='qwen2.5:1.5b', messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        content = response['message']['content'].replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        # Ensure recommended_mode exists
        if "recommended_mode" not in result:
            risk = result.get("risk_intent", "NONE")
            if risk == "CRISIS": result["recommended_mode"] = "REFERRAL"
            elif risk == "DISTRESS": result["recommended_mode"] = "SUPPORT"
            else: result["recommended_mode"] = "THERAPY"

        return result
    
    except Exception as e:
        print(f"⚠️ Local Risk error: {e}")
        # Default to safe mode (DISTRESS/SUPPORT)
        return {
            "risk_intent": "DISTRESS",
            "confidence": 0.5,
            "recommended_mode": "SUPPORT"
        }

# ============================================================
# ========= STAGE 4: CONVERSATION SYNTHESIS ===================
# ============================================================

def generate_system_prompt(risk_mode):
    """Generate appropriate system prompt based on risk level"""
    
    base = """You are clinical, a compassionate AI mental health support assistant. You provide evidence-based guidance while maintaining warmth and empathy.

Core Principles:
- Always validate emotions before offering solutions
- Use active listening and reflective responses
- Never diagnose conditions or prescribe medication
- Suggest professional help when appropriate
- Use evidence-based therapeutic techniques (CBT, DBT, ACT, MI)"""
    
    if risk_mode == "CRISIS":
        return base + """

⚠️ CRISIS MODE - This user may be in immediate danger.

Your response MUST:
1. Acknowledge their pain with urgency and compassion
2. Immediately provide: 988 Suicide & Crisis Lifeline (24/7)
3. Suggest calling emergency services (911) if in immediate danger
4. Validate feelings while strongly emphasizing safety
5. Ask if they have a support person nearby
6. Do not try to "solve" the crisis - focus on immediate safety"""
    
    elif risk_mode == "DISTRESS":
        return base + """

⚠️ DISTRESS MODE - User experiencing significant emotional pain.

Prioritize:
1. Deep emotional validation ("That sounds incredibly hard")
2. Grounding techniques if overwhelmed
3. Immediate coping strategies (breathing, self-care)
4. Assessment of support system
5. Gentle exploration of what might help right now"""
    
    else:
        return base + """

Engage in therapeutic conversation:
- Validate emotions authentically
- Ask open-ended clarifying questions
- Suggest appropriate assessments when symptoms are clear
- Provide psychoeducation when helpful
- Encourage small, actionable steps"""

def synthesize_training_conversations(chunk, channel_meta, delivery_style, risk_mode):
    """
    Generate realistic therapy conversation pairs for fine-tuning
    
    Args:
        chunk: Text chunk from transcript
        channel_meta: Channel metadata dict
        delivery_style: Detected delivery style
        risk_mode: Risk classification
    
    Returns:
        list: Training examples in LLaMA format
    """
    
    system_prompt = f"""
You are generating high-quality training data for clinical, a mental health AI assistant.

TASK: Create 2-3 realistic therapy conversation pairs based on this source material.

SOURCE MATERIAL: {chunk}

CONTEXT:
- Therapist: {channel_meta['name']} ({channel_meta['credentials']})
- Delivery Style: {delivery_style}
- Risk Level: {risk_mode}
- Quality Tier: {channel_meta['quality_tier']}

OUTPUT FORMAT (strict JSON):
{{
  "conversations": [
    {{
      "user": "What a real user might say (1-3 sentences)",
      "assistant": "How clinical should respond (3-5 sentences)",
      "metadata": {{
        "intervention": "One of: {', '.join(CANONICAL_INTERVENTIONS)}",
        "risk_level": "none|distress|crisis",
        "therapeutic_technique": "CBT|DBT|ACT|MI|Validation|etc"
      }}
    }}
  ]
}}

CRITICAL RULES:
1. User messages should be natural, conversational, 1-3 sentences
2. Assistant responses should be 3-5 sentences (not too long)
3. Use the {delivery_style} style consistently
4. If CRISIS mode: MUST include 988 hotline reference in assistant response
5. NO medical diagnosis (never say "you have X disorder")
6. NO medication advice (never suggest drugs or dosages)
7. Validate emotions before offering solutions
8. Ask clarifying questions to deepen understanding
9. Generate 2-3 diverse conversation pairs

EXAMPLES OF GOOD STYLE:

Warm & Maternal:
User: "I feel like I'm drowning in everything."
Assistant: "Oh honey, that sounds so overwhelming. When you say drowning, I'm hearing that it feels like too much all at once. Can you tell me what's been piling up? Sometimes naming the pieces helps us see what we can actually manage."

Direct & Coach-like:
User: "I keep saying I'll exercise but I never do."
Assistant: "Alright, let's get real here. You know what you need to do, so what's actually getting in the way? Is it time, energy, motivation, or something else? We need to identify the real blocker before we can tackle it."

Quiet & Contemplative:
User: "I don't know who I am anymore."
Assistant: "That's a profound thing to notice about yourself. When did you first start feeling this shift? Sometimes we lose ourselves gradually, and pausing to witness that takes real courage. What parts of yourself do you remember most clearly?"
"""
    
    try:
        msg = claude.messages.create(
            # CHANGED: Switched to Haiku (12x cheaper)
            model="claude-3-haiku-20240307", 
            max_tokens=3000,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": chunk[:6000]}] 
        )
        
        # Log cost
        cost_tracker.log_claude(
            msg.usage.input_tokens,
            msg.usage.output_tokens,
            model="haiku"  # <--- IMPORTANT: Update this to "haiku" for correct reporting
        )
        
        result = json.loads(clean_json(msg.content[0].text))
        
        # Convert to LLaMA format
        training_examples = []
        for i, conv in enumerate(result.get("conversations", [])):
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": generate_system_prompt(conv["metadata"]["risk_level"])
                    },
                    {
                        "role": "user",
                        "content": conv["user"]
                    },
                    {
                        "role": "assistant",
                        "content": conv["assistant"]
                    }
                ],
                "metadata": {
                    "intervention": conv["metadata"]["intervention"],
                    "risk_level": conv["metadata"]["risk_level"],
                    "therapeutic_technique": conv["metadata"].get("therapeutic_technique", "General"),
                    "delivery_style": delivery_style,
                    "source_channel": channel_meta["name"],
                    "quality_tier": channel_meta["quality_tier"]
                }
            }
            
            # Quality check before adding
            if quality_check(example):
                training_examples.append(example)
                cost_tracker.costs["total_conversations"] += 1
        
        return training_examples
    
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return []

# ============================================================
# ========= STAGE 5: QUALITY CONTROL ==========================
# ============================================================

def flag_for_manual_review(example, reason):
    """Flag examples for human review"""
    with open(REVIEW_QUEUE, "a") as f:
        f.write(json.dumps({
            "example": example,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "reviewed": False
        }) + "\n")

def quality_check(training_example):
    """
    Automated quality checks for safety and appropriateness
    
    Returns:
        bool: True if example passes all checks
    """
    try:
        assistant_msg = training_example["messages"][2]["content"]
    except (IndexError, KeyError):
        return False
    
    assistant_lower = assistant_msg.lower()
    
    # Check 1: No medical diagnosis
    diagnosis_patterns = [
        r"you (have|are|suffer from|are diagnosed with)",
        r"you('re| are) (definitely|clearly|obviously)",
        r"this (is|sounds like) (depression|anxiety|bipolar|schizophrenia|adhd)",
        r"you meet the criteria for"
    ]
    for pattern in diagnosis_patterns:
        if re.search(pattern, assistant_lower):
            flag_for_manual_review(training_example, f"Diagnosis language: {pattern}")
            return False
    
    # Check 2: No medication advice
    medication_keywords = [
        "take this medication",
        "stop taking",
        "increase dosage",
        "decrease dosage",
        "you should take",
        "prescription",
        "mg of",
        "antidepressant",
        "ssri",
        "benzodiazepine"
    ]
    for keyword in medication_keywords:
        if keyword in assistant_lower:
            flag_for_manual_review(training_example, f"Medication advice: {keyword}")
            return False
    
    # Check 3: Response length (not too short/long)
    word_count = len(assistant_msg.split())
    if word_count < 10:
        return False  # Too terse
    if word_count > 200:
        flag_for_manual_review(training_example, f"Response too long: {word_count} words")
        return False
    
    # Check 4: Must have some empathy/validation
    empathy_indicators = [
        "that sounds", "i hear", "that must", "i can imagine",
        "makes sense", "understandable", "valid to feel",
        "that's hard", "that's tough", "sounds difficult"
    ]
    has_empathy = any(phrase in assistant_lower for phrase in empathy_indicators)
    if not has_empathy and training_example["metadata"]["risk_level"] != "none":
        flag_for_manual_review(training_example, "Lacks empathy/validation")
        return False
    
    # Check 5: Crisis mode must include resources
    if training_example["metadata"]["risk_level"] == "crisis":
        has_988 = "988" in assistant_msg
        has_crisis_resource = any(word in assistant_lower for word in ["hotline", "crisis", "emergency"])
        if not (has_988 or has_crisis_resource):
            flag_for_manual_review(training_example, "Crisis mode but no resources provided")
            return False
    
    # Check 6: Random sampling for manual review (1% of all data)
    if random.random() < 0.01:
        flag_for_manual_review(training_example, "Random sample for quality audit")
    
    return True

# ============================================================
# ========= STAGE 6: ROUTING & DEDUPLICATION ==================
# ============================================================

def save_training_example(example, video_id, chunk_idx, variation_idx):
    """
    Save training example with deduplication
    
    Args:
        example: Training example dict
        video_id: Source video ID
        chunk_idx: Chunk index
        variation_idx: Variation number
    """
    # Generate hash for deduplication
    hash_id = compute_hash(video_id, chunk_idx, variation_idx)
    
    if hash_id in SEEN_HASHES:
        return  # Already processed
    
    # Add hash to example
    example["hash"] = hash_id
    example["source"] = {
        "video_id": video_id,
        "chunk_index": chunk_idx,
        "variation_index": variation_idx,
        "processed_at": datetime.utcnow().isoformat()
    }
    
    # Save to training file
    with open(TRAINING_DATA, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Mark as seen
    mark_seen(hash_id, {
        "video_id": video_id,
        "chunk_idx": chunk_idx,
        "variation_idx": variation_idx
    })

# ============================================================
# ========= STAGE 7: YOUTUBE VIDEO FETCHING ===================
# ============================================================

def get_channel_videos(channel_id):
    """
    Fetch all videos from a YouTube channel
    
    Args:
        channel_id: YouTube channel ID
    
    Returns:
        list: Video metadata dicts
    """
    videos = []
    
    try:
        # PATCH 3: YouTube Items Guard
        channel_response = youtube.channels().list(
            id=channel_id,
            part="contentDetails"
        ).execute()
        
        items = channel_response.get("items", [])
        if not items:
            print(f"⚠️ Channel {channel_id} has no accessible uploads playlist")
            return []
        
        uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Paginate through all videos
        next_page_token = None
        
        while True:
            playlist_response = youtube.playlistItems().list(
                playlistId=uploads_playlist,
                part="snippet",
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            for item in playlist_response["items"]:
                video_id = item["snippet"]["resourceId"]["videoId"]
                title = item["snippet"]["title"]
                description = item["snippet"].get("description", "")
                
                # Filter non-therapeutic content
                if not is_therapeutic_content(title, description):
                    cost_tracker.costs["filtered_videos"] += 1
                    continue
                
                videos.append({
                    "id": video_id,
                    "title": title,
                    "description": description,
                    "channel_id": channel_id,
                    "channel_meta": CHANNEL_METADATA.get(channel_id, {})
                })
            
            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break
            
            time.sleep(0.5)  # Rate limiting
        
        return videos
    
    except Exception as e:
        print(f"❌ Error fetching videos for channel {channel_id}: {e}")
        return []

# ============================================================
# ====================== MAIN PIPELINE ========================
# ============================================================
def main():
    """Main pipeline execution"""
    
    print("="*70)
    print("🔥 clinical YouTube Training Data Generator v4.2")
    print("="*70)

    print(f"\n📊 Configuration:")
    print(f"  • Channels: {len(CHANNEL_IDS)}")
    print(f"  • Chunk Size: {CHUNK_SIZE} characters")
    print(f"  • Rate Limit: {RATE_LIMIT_SECONDS}s between API calls")
    print(f"\n📁 Output Files:")
    print(f"  • Training Data: {TRAINING_DATA}")
    print(f"  • Review Queue: {REVIEW_QUEUE}")
    print(f"  • Cost Report: {COST_REPORT}")
    print("\n" + "="*70 + "\n")

    for channel_id in CHANNEL_IDS:
        channel_meta = CHANNEL_METADATA.get(channel_id, {})
        channel_name = channel_meta.get("name", channel_id)
        
        print(f"\n📺 Processing Channel: {channel_name}")
        print(f"   Credentials: {channel_meta.get('credentials', 'Unknown')}")
        print(f"   Quality Tier: {channel_meta.get('quality_tier', 'Unknown')}")
        print("   " + "-"*60)
        
        videos = get_channel_videos(channel_id)
        print(f"   ✅ Found {len(videos)} therapeutic videos")
        
        for video in tqdm(videos, desc=f"   Videos ({channel_name})", unit="video"):
            cost_tracker.costs["total_videos"] += 1
            
            result = download_and_transcribe(video["id"])
            if not result["success"]:
                continue
            
            transcript = result["text"]
            chunks = textwrap.wrap(transcript, CHUNK_SIZE)
            
            for chunk_idx, chunk in enumerate(chunks):
                # ⏳ PACING: Sleep at the START of the loop (except for the first chunk)
                # This prevents hammering the API if we hit 'continue' below.
                if chunk_idx > 0:
                    time.sleep(RATE_LIMIT_SECONDS)

                cost_tracker.costs["total_chunks"] += 1
                
                # 1. Local Gate (Free)
                relevance = gate_relevance_and_style(video["title"], chunk)
                if not relevance.get("relevant") or relevance.get("confidence", 0) < 0.6:
                    print(f"   ⏭️ Chunk {chunk_idx} skipped (Irrelevant or Low Confidence)")
                    continue
                
                # 2. Local Risk (Free)
                risk = classify_risk_intent(chunk)
                
                # 3. Claude Synthesis (Paid - High Quality)
                conversations = synthesize_training_conversations(
                    chunk,
                    video["channel_meta"],
                    relevance.get("delivery_style", "Warm & Maternal"),
                    risk.get("recommended_mode", "SUPPORT")
                )
                
                for variation_idx, example in enumerate(conversations):
                    save_training_example(
                        example,
                        video["id"],
                        chunk_idx,
                        variation_idx
                    )
                
                # (Removed sleep from bottom)
                
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    cost_tracker.report()


if __name__ == "__main__":
    main()