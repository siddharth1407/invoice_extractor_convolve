# Tractor Invoice Information Extraction System

## 1. Executive Summary
This submission implements a high-precision extraction pipeline for analyzing Tractor Invoices. The system is designed to handle the linguistic diversity of Indian invoices (English, Hindi, Marathi, Gujarati) and unstructured layouts using a Vision-Language Model (VLM) approach.

## 2. System Architecture

### 2.1 Model Selection
We utilized **Qwen2-VL-2B-Instruct** for inference.
- **Rationale:** The 2B parameter count offers the optimal trade-off between semantic understanding and inference latency.
- **Precision:** Model is loaded in `bfloat16` to optimize memory usage on consumer hardware (16GB RAM constraint) without sacrificing accuracy.

### 2.2 Inference Pipeline
1.  **Preprocessing:** Images are dynamically resized (Max 1024px) to preserve text clarity while minimizing token count.
2.  **Zero-Shot CoT (Chain of Thought):** A strict system prompt enforces "Vernacular Fidelity," preventing the model from translating local scripts into English.
3.  **Hybrid Extraction Logic:**
    * *Primary:* VLM extraction via JSON generation.
    * *Secondary:* Regex-based fallback logic for extracting `Horse Power` from unstructured model descriptions.
    * *Validation:* Range checks (e.g., HP must be 30-90) to reject model number hallucinations.

## 3. Cost & Performance Analysis

### 3.1 Latency
- **Average Processing Time:** ~18 seconds per document (on Apple M4 / MPS).
- **Optimization:** Gradient calculation disabled (`torch.no_grad`) and resolution capping ensures steady throughput.

### 3.2 Cost Estimate
The cost estimate in the output JSON (`$0.002`) is derived from equivalent Cloud GPU pricing:
- **Instance:** NVIDIA T4 or A10G equivalent.
- **Throughput:** ~3-4 docs/minute.
- **Hourly Cost:** ~$0.35/hr.
- **Per Document:** ~$0.0015 - $0.002.

This architecture is significantly cheaper (>10x) than using closed-source APIs like GPT-4o or Gemini 1.5 Pro.

## 4. Key Features
- **Vernacular Support:** Native script retention (Hindi/Marathi) without hallucinated translation.
- **Visual Grounding:** Extracts bounding boxes (`bbox`) for signatures and stamps for fraud detection.
- **Robustness:** Handles noisy OCR contexts via fuzzy matching and regex post-processing.

## 5. How to Run

### 1. Prepare Input Data
- Insert your test images (`PNG`, `JPG`, `JPEG`, or `PDF`) into input_images folder.

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# On Windows:
# venv\Scripts\activate

3. **Install dependencies:**
    ```bash
   pip install -r requirements.txt

4. **Run the executable:**
    ```bash
    python executable.py --input_dir input_images --output_file result.json
    
