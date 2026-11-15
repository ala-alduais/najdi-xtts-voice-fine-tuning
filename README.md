# 🗣️ Najdi Arabic Voice Cloning with XTTS-v2

Fine-tuning **XTTS-v2** on ~1.7 hours of my own recorded speech to generate natural **Saudi Najdi Arabic** in my voice, trained entirely on a free Kaggle Tesla T4.

> **Result:** zero-shot XTTS-v2 cloned my voice but sounded robotic and not Najdi. After fine-tuning, a blind listening test showed clear gains in **Najdi dialect authenticity** and **voice similarity**, and the longest mid-sentence pauses dropped from 1.35 s to 0.77 s on average. A second experiment (v2) removed recording hesitations from the training data, which fixed the remaining unnatural pauses. **v2 was preferred in all 10 sentences** of a blind A/B/C test against the base model and v1.

---

## Table of contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Data pipeline](#data-pipeline)
- [Model & training](#model--training)
- [Experiments](#experiments)
- [Evaluation](#evaluation)
- [Engineering challenges](#engineering-challenges)
- [Limitations](#limitations)
- [Reproducing this project](#reproducing-this-project)
- [Responsible use & license](#responsible-use--license)

---

## Motivation

Most Arabic TTS models are trained on Modern Standard Arabic. Saudi dialects such as **Najdi** have their own pronunciation, rhythm and vocabulary, and pretrained models usually sound formal or "foreign" when asked to speak them. This project tests how far a strong multilingual model can be adapted to one speaker and one dialect using a small, self-recorded dataset and zero budget.

**Earlier attempt:** I first tried fine-tuning Spark-TTS-0.5B on a public Najdi dataset. The pipeline worked end to end, but the model had no Arabic in its pretraining and the speech stayed unintelligible. I switched to XTTS-v2, which already supports Arabic, so fine-tuning only has to adapt the dialect and the voice.

---

## Dataset

Recorded by me on a phone, reading prepared Najdi sentences (bookings, orders, Saudi names).

| Property | Value |
|---|---|
| Recordings planned / usable | 1,810 / **756** (the remaining audio files were lost during export) |
| Format | 48 kHz, mono, 32-bit PCM WAV |
| Duration (after silence trimming) | **1.71 h** |
| Clip length | 5.2 – 18.5 s (median 8.6 s) |
| Recording sessions | 15 sessions over 2 weeks |
| Text | 40 – 87 characters, no digits or Latin script |

**Audit results:** 0 unreadable files, 0 silent clips, 0 clipped clips, and 0 duration mismatches between the audio files and the CSV metadata (a check that each audio file belongs to its transcript).

> The dataset is not published because it is a recording of my own voice.

---

## Data pipeline

1. **Match** CSV rows to the audio files that exist (756 clips).
2. **Convert** 48 kHz → 22,050 Hz mono, 16-bit PCM (the format XTTS expects). No denoising and no loudness normalization, to preserve voice quality.
3. **Trim** leading and trailing silence (`top_db=40`, 150 ms padding kept).
4. **Compress internal pauses** (v2 only): silent gaps longer than 0.3 s inside a sentence are shortened to 0.3 s. Cuts happen only inside silence.
5. **Split by recording session**, not randomly. Clips from the same sitting share mic position and room sound, so a random split would leak into validation and test.

| Split | Sessions | Clips | Minutes (v1) |
|---|---|---|---|
| Train | 0–12 | 701 | 92.6 |
| Validation | 13 | 35 | 6.2 |
| Test | 14 | 20 | 3.8 |

---

## Model & training

XTTS-v2 has three parts: a DVAE audio tokenizer, a **GPT model** that predicts audio tokens from text, and a HiFi-GAN decoder. Only the **GPT** part is fine-tuned. It controls pronunciation, rhythm and dialect, while the frozen decoder preserves audio quality. This is also what makes training fit on a 16 GB T4.

| Setting | Official recipe | This project | Reason |
|---|---|---|---|
| Batch × grad accumulation | 3 × 84 | **2 × 1** | With 701 clips, an effective batch of 252 gives fewer than 3 updates per epoch |
| Max audio length | 11.6 s | **19 s** | Keeps the 138 longer clips (18% of the data) |
| Learning rate | 5e-6 | 5e-6 | Low to avoid damaging the pretrained voice |
| Checkpointing | every 10k steps | **best by validation loss** | Small dataset overfits quickly, and disk is limited |
| Precision | fp32 | fp32 | T4 has no bf16 |

**Hardware:** Kaggle Tesla T4 (16 GB), about 4.5 minutes per epoch.

### Training curve (v1)

| Epoch | Validation mel loss |
|---|---|
| 0 (pretrained) | 3.863 |
| 1 | 3.217 |
| 2 | 3.147 |
| 4 | 3.129 |
| **6** | **3.113** ← best checkpoint |
| 7 | 3.133 |
| 9 | 3.154 |

Training loss kept decreasing (to about 2.55), while validation loss reached its minimum at **epoch 6** and then rose. That is **overfitting**, so the epoch-6 checkpoint was kept.

---

## Experiments

| Version | Change | Motivation |
|---|---|---|
| **Base** | Pretrained XTTS-v2, zero-shot cloning from 3 reference clips | Baseline |
| **v1** | Fine-tuned GPT on my recordings | Adapt dialect and voice |
| **v2** | v1 + internal pause compression in the training data | Fix mid-sentence pauses learned from my recordings |

**Why v2:** v1 sometimes paused mid-sentence and then continued. Measuring my training data showed the cause:

- **54%** of training clips contained a pause longer than 0.5 s
- **18%** contained a pause longer than 1.0 s

These were reading hesitations, for example before a name. The model learned them as part of my speaking style. Compressing them shortened 58% of clips and removed 7.7 minutes of silence, with no words lost (checked by listening).

---

## Evaluation

All models were compared under identical conditions: the same 3 reference clips, the same seed, and the same 10 sentences:

- **1–5:** held-out test sentences from an unseen recording session (with my real recording for comparison)
- **6–8:** casual conversational Najdi, a different style from the training sentences
- **9–10:** sentences containing **Saudi names not in the dataset**

### Blind listening test
Versions were presented in random order (A/B/C) and scored before the mapping was revealed.

| Criterion | Base | v1 | v2 |
|---|---|---|---|
| Voice similarity | Good | Better | Better |
| Najdi dialect | Robotic, not Najdi | Clearly more Najdi | Clearly more Najdi |
| Naturalness | Robotic | Improved, but some mid-sentence pauses | **Natural, pauses fixed** |
| Name pronunciation | Good | Good | Good |
| Preferred in (of 10 sentences) | 0 | 0 | **10** |

v2 was preferred in every sentence, including the conversational sentences and the sentences with unseen names. Removing hesitations from the training data was the change that made the output sound natural.

### Objective pause measurement
Longest silent gap inside each generated sentence (`librosa.effects.split`, `top_db=35`):

| Model | Avg. longest pause | Sentences with a pause > 0.8 s |
|---|---|---|
| Base | 1.35 s | 8 / 10 |
| v1 | 0.77 s | 4 / 10 |

No generated sentence was cut off: v1 durations were 0.90–1.27× the duration expected from my real speaking rate.

---

## Engineering challenges

| Problem | Cause | Fix |
|---|---|---|
| Only 804 of 1,810 audio files available | Incomplete export from the recording platform | Audited CSV ↔ audio matches and continued with the 756 verified pairs |
| `ImportError: XttsAudioConfig` | Class moved in newer `coqui-tts` | Import from `TTS.tts.models.xtts` |
| `load_tts_samples` API error | Function now expects the full config object | Pass `datasets=[...]` inside the config |
| Trainer refused to start | Kaggle session had 2 GPUs | `CUDA_VISIBLE_DEVICES=0` |
| Models loaded as `None` | `Xtts.eval()` returns `None` | Call `.cuda()` without chaining `.eval()` |
| Trained model lost | Kaggle deletes `/kaggle/working` when an idle session ends | Train in commit mode and export only the weights (no optimizer state) |
| Mid-sentence pauses | Hesitations in the training recordings | Pause compression (v2) |

---

## Limitations

- **Small dataset:** 1.7 h, and only 756 of the 1,810 planned recordings were usable.
- **Limited text variety:** mostly templated sentences (bookings, names), so casual conversational Najdi improves less than in-domain sentences.
- **Phone recordings** in a non-studio environment.
- **Single listener:** the blind test was scored by me. A proper MOS study needs multiple native Najdi listeners.
- **No dedicated Najdi language code:** XTTS only has a general `ar` code.

**Future work:** record conversational sentences, recover the missing recordings, add speech-recognition error rate (CER) and speaker-similarity metrics, and run a multi-listener MOS study.

---

## Reproducing this project

Everything runs on **Kaggle** (free GPU T4, Internet enabled).

```
notebooks/
├── train_v1.ipynb                   # data prep + training + export (v1)
├── train_v2_pause_compression.ipynb # same with internal pause compression (v2)
└── evaluate.ipynb                   # base vs v1 vs v2: blind test + pause metrics
```

1. Upload your dataset (`metadata.csv` with `Text`, `audio_path`, `recorded_at`, `id`, plus a `wavs/` folder) as a Kaggle dataset.
2. Run a training notebook with **Save Version → Save & Run All (Commit)**, so it survives idle timeouts and saves the model as notebook output.
3. Attach the training notebook's output to `evaluate.ipynb` and run it.

**Main dependencies:** `coqui-tts` 0.28, `coqui-tts-trainer`, PyTorch 2.11, `librosa`, `soundfile`, `pandas`.

---

## Responsible use & license

- This project clones **my own voice** only. Do not use voice cloning on anyone without their explicit consent.
- The fine-tuned weights are **not published**, since they would let anyone generate speech in my voice.
- **XTTS-v2 weights** are released under the [Coqui Public Model License](https://coqui.ai/cpml), which permits **non-commercial use only**. Any model fine-tuned from them carries the same restriction.
- Code in this repository: MIT License.

## Acknowledgments
- [Coqui TTS / XTTS-v2](https://github.com/idiap/coqui-ai-TTS) (maintained fork by Idiap)
- Kaggle for free GPU compute
