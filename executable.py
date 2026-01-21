import os
import json
import torch
import time
import ast
import argparse
import gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.processing import clean_cost, smart_extract_hp, calculate_confidence, validate_bbox

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

SYSTEM_INSTRUCTION = """
You are an expert OCR extraction engine for Indian Invoices. 
Your task is to extract specific fields EXACTLY as they appear in the document.

CRITICAL RULES:
1. **LANGUAGE FIDELITY**: If text is in Hindi/Marathi (e.g., "श्री गणेश"), output "श्री गणेश". DO NOT TRANSLATE to English.
2. **NUMERIC PRECISION**: Extract integers for Cost and HP.
3. **FORMAT**: Output strictly valid JSON.

FIELDS TO EXTRACT:
1. dealer_name: Name of dealer (Header/Stamp). Keep original script.
2. model_name: Full Tractor Model description.
3. horse_power: The HP number (15-90). Look for 'HP'.
4. asset_cost: Grand Total / Net Amount.
5. signature: Bounding box [ymin, xmin, ymax, xmax].
6. stamp: Bounding box [ymin, xmin, ymax, xmax].
"""


def load_model():
    """Initializes the Qwen2-VL Model and Processor."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print(f"🚀 Loading Inference Engine on {device.upper()}...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=256*28*28, max_pixels=1024*28*28
    )

    return model, processor, device


def process_single_image(model, processor, device, image_path, filename):
    start_time = time.time()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": SYSTEM_INSTRUCTION}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_ids = outputs.sequences
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(
        inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True)[0]

    confidence_score = calculate_confidence(outputs.scores)
    processing_time = round(time.time() - start_time, 2)

    try:

        clean_text = output_text.replace(
            "```json", "").replace("```", "").strip()
        if "{" in clean_text:
            clean_text = clean_text[clean_text.find(
                "{"):clean_text.rfind("}")+1]

        try:
            fields = json.loads(clean_text)
        except:
            fields = ast.literal_eval(clean_text)

        if "asset_cost" in fields:
            fields["asset_cost"] = clean_cost(fields["asset_cost"])

        fields["horse_power"] = smart_extract_hp(fields)

        if "dealer_name" in fields and isinstance(fields["dealer_name"], str):
            fields["dealer_name"] = fields["dealer_name"].replace(
                "\n", " ").strip()

        fields["signature"] = validate_bbox(fields.get("signature"))
        fields["stamp"] = validate_bbox(fields.get("stamp"))

        return {
            "doc_id": filename,
            "fields": fields,
            "confidence": confidence_score,
            "processing_time_sec": processing_time,
            "cost_estimate_usd": 0.002
        }

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return {
            "doc_id": filename,
            "fields": {},
            "confidence": 0.0,
            "processing_time_sec": processing_time,
            "cost_estimate_usd": 0.000,
            "error": "Extraction Failed"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Invoice Extraction Executable")
    parser.add_argument("--input_dir", type=str,
                        default="input_images", help="Path to input images")
    parser.add_argument("--output_file", type=str,
                        default="sample_output/result.json", help="Path to output JSON")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model, processor, device = load_model()

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.pdf'))]
    files.sort()

    print(f"📸 Found {len(files)} documents in {args.input_dir}")

    results = []

    for i, filename in enumerate(files):
        path = os.path.join(args.input_dir, filename)
        result = process_single_image(model, processor, device, path, filename)
        results.append(result)

        if i % 10 == 0:
            torch.mps.empty_cache()
            gc.collect()

        f_data = result.get("fields", {})
        print(f"[{i+1}/{len(files)}] {filename} | {f_data.get('dealer_name')} | ₹{f_data.get('asset_cost')} | Conf: {result.get('confidence')}")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Processing Complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
