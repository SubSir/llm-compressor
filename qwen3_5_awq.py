import os
from huggingface_hub import HfApi

from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

def main(model_name: str = "Qwen/Qwen3.5-30B-A3B"):
    """
    Run AWQ calibration on Qwen3.5 models and upload to HuggingFace.
    
    Args:
        model_name: HuggingFace model name to quantize
    """
    # Select calibration dataset.
    DATASET_ID = "mit-han-lab/pile-val-backup"
    DATASET_SPLIT = "validation"

    # Select number of samples. 256 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 128
    MAX_SEQUENCE_LENGTH = 512

    # Select model and load it.
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        if "messages" in example:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        else:
            text = example.get("text") or example.get("content") or ""
        return {"text": text}

    ds = ds.map(preprocess)

    # Tokenize inputs.
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    # Configure the quantization algorithm to run.
    recipe = [
        AWQModifier(
            ignore=["lm_head", "re:.*linear_attn.in_proj_b$", "re:.*linear_attn.in_proj_a$"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
            duo_scaling="both",
        ),
    ]

    # Apply algorithms.
    print("Applying AWQ quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,  # type: ignore[arg-type]
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    from llmcompressor.pytorch.utils.helpers import get_quantized_layers  
  
    quantized_layers = get_quantized_layers(model)  
    for name, module in quantized_layers:  
        print(f"Quantized: {name} {module}")

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save both variants to disk: legacy fake (non-compressed) and real AWQ (compressed).
    model_suffix = model_name.rstrip("/").split("/")[-1]
    fake_save_dir = model_suffix + "-awq-asym-fake"
    # real_save_dir = model_suffix + "-awq-asym"

    print(f"Saving fake quantized model to: {fake_save_dir}")
    model.save_pretrained(fake_save_dir, save_compressed=False)
    tokenizer.save_pretrained(fake_save_dir)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AWQ calibration for Qwen3.5 models")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model name to quantize"
    )
    args = parser.parse_args()
    
    main(model_name=args.model_name)
