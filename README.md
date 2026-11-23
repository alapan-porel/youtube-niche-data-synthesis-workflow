# Clinical YouTube Niche Data Factory

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![yt-dlp](https://img.shields.io/badge/Audio-yt--dlp-red.svg)](https://github.com/yt-dlp/yt-dlp)
[![Whisper](https://img.shields.io/badge/Transcription-Faster--Whisper-teal.svg)](https://github.com/SYSTRAN/faster-whisper)

An industrial data-engineering pipeline harvesting, transcribing, and chunking clinical psychology and psychiatric YouTube video feeds. Extracts spoken conversational wisdom and therapy roleplays from vetted licensed therapists into structured synthetic fine-tuning corpora.

---

## Architectural Pipeline

```
 [ Curated Clinical Channels ]
               │
               ▼
   ┌──────────────────────────────────────────────┐
   │ 1. Channel Scanner (YouTube API v3)          │
   │    ├─ Filters clinical topics (CBT, OCD, ACT)│
   │    └─ Evaluates speaker credentials & tenure │
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │ 2. Media Downloader (yt-dlp)                 │
   │    ├─ Headless audio extraction (16kHz mono) │
   │    └─ Automatic retry & rate-limit throttling│
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │ 3. Speech-to-Text (Faster-Whisper GPU)       │
   │    ├─ Timestamped word-level segmentation    │
   │    └─ Speaker diarization & semantic chunking│
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │ 4. Corpus Normalization & Deduplication      │
   │    ├─ SHA-256 chunk deduplication            │
   │    └─ Brand & PII sanitization               │
   └──────────────────────────────────────────────┘
```

---

## Channel Registry Taxonomy

The factory operates strictly on verified, licensed institutional and practitioner channels. Channels are indexed with clinical specializations:
* **Child & Adolescent Psychology** (e.g. Harvard Graduate School of Education)
* **Trauma & Somatic Recovery** (e.g. Licensed Clinical Social Workers)
* **Anxiety & Exposure Therapy** (e.g. DARE Response, OCD Specialists)

---

## Installation & Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Set your YOUTUBE_API_KEYS in .env
```

### Running Channel Scanner

```bash
python src/channel_scanner.py
```

### Running Pipeline Ingestion

```bash
python src/pipeline_runner.py
```

---

## Datasets

* **25-row sample**: `samples/clinical_transcripts_sample.jsonl`
* **Full transcript corpus**: `_huggingface_datasets/clinical_youtube_transcripts_merged.jsonl` (592 deduplicated clinical audio chunks extracted from 276 educational videos).

---

## License

Apache-2.0. Cleanroom extraction and packaging release.
