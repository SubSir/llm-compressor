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
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # Select number of samples. 256 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 256
    MAX_SEQUENCE_LENGTH = 512

    # Select model and load it.
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

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

    # Save to disk compressed.
    save_dir = model_name.rstrip("/").split("/")[-1] + "-awq-asym"
    print(f"Saving quantized model to: {save_dir}")
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

    # Upload to HuggingFace
    repo_name = f"SubSir/{model_name.split('/')[-1]}-AWQ"
    print(f"Uploading to HuggingFace: {repo_name}")
    
    api = HfApi()
    api.create_repo(repo_id=repo_name, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_name,
        repo_type="model",
    )
    
    print(f"Successfully uploaded to https://huggingface.co/{repo_name}")
    
    return {"model_path": save_dir, "repo_id": repo_name}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AWQ calibration for Qwen3.5 models")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="HuggingFace model name to quantize"
    )
    args = parser.parse_args()
    
    main(model_name=args.model_name)
