# SkinGPT

An agentic dermatology assistant. You upload a photo of a skin condition. A custom-trained YOLOv5 model detects it, a visual memory store finds similar past cases, and a ReAct-style LLM agent reasons over the results. The agent can ask follow-up questions, write a structured diagnostic report, or answer questions in a chat.

> ⚠️ Research and educational project only. It is not a medical device and does not replace a dermatologist.

## Pipeline

```
Image upload (Flask)
      │
      ▼
YOLOv5 detector (best.pt) ──► top disease + confidence
      │
      ▼
ResNet-18 image embedding ──► ChromaDB (cosine) ──► similar past cases
      │                              ▲
      │                              └── diagnosis stored for future lookups
      ▼
ReAct agent (Llama 3.1 8B via Ollama)
  THOUGHT → ACTION → OBSERVATION  (up to 5 iterations)
  tools: SEARCH_MEMORY · ASK_FOLLOWUP · GENERATE_REPORT · ANSWER
      │
      ▼
Chat UI: structured report, follow-up questions, agent reasoning trace
```

## Detectable conditions

The YOLOv5 model (`best.pt`) was trained on 12 classes:

Acne · Chickenpox · Eczema · Monkeypox · Pimple · Psoriasis · Ringworm · Basal cell carcinoma · Melanoma · Tinea versicolor · Vitiligo · Warts

## Components

| File | Role |
|---|---|
| `app.py` | Flask app. `/upload` runs detection, memory lookup and the agent. `/query` handles chat turns and answers to follow-up questions. |
| `yolo.py` | Loads the custom YOLOv5 weights and returns the detection with the highest confidence |
| `memory.py` | Visual memory: 512-dim ResNet-18 embeddings stored in a persistent ChromaDB collection (`./chroma_db`), used for similar-case retrieval |
| `orchestrator.py` | ReAct agent loop: parses `THOUGHT/ACTION/INPUT`, dispatches tools, and pauses on `ASK_FOLLOWUP` until the user replies |
| `report.py` | Generates the structured markdown report and a rule-based severity estimate (Severe / Moderate / Mild) |
| `templates/` | Upload page, chat interface and response views |

## Getting started

**Prerequisites:** Python 3.9+ and [Ollama](https://ollama.com) running locally.

```bash
# 1. LLM
ollama pull llama3.1:8b

# 2. YOLOv5 source (yolo.py loads it from ./yolov5)
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt

# 3. App dependencies
pip install flask torch torchvision pillow chromadb requests

# 4. Run
python app.py      # http://127.0.0.1:5000
```

Upload an image (there's a sample in `uploads/`). You'll get the detection, similar cases and the agent's first response, then you can keep chatting.

## Tech stack

Python · Flask · PyTorch · YOLOv5 · torchvision ResNet-18 · ChromaDB · Ollama (Llama 3.1 8B) · ReAct agent pattern

## Roadmap

This version is the YOLOv5-based prototype. Planned next steps include moving to vision-language models (a fine-tuned BiomedCLIP classifier compared against a zero-shot LLaVA-Med baseline), a tiered memory architecture, and LoRA/QLoRA fine-tuning.
