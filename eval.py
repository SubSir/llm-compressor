# eval.py
import torch
import random
import argparse
import subprocess
from tqdm import tqdm
from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
from tqdm import tqdm

def load_model(
    model_path: str, device_map: str | None = None, dtype=torch.float32, **kwargs
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=dtype, **kwargs
    )
    return model

def load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_wikitext2(seed, seqlen, tokenizer):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return testenc.input_ids


def get_wikitext2_trainloader(seed, seqlen, tokenizer, nsamples):
    random.seed(seed)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    traindata = traindata.filter(lambda x: len(x) > 0)
    traindata = traindata.map(lambda x: {"text": x["text"].strip()})
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_c4(seed, seqlen, tokenizer):
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    random.seed(seed)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return valenc


def get_test_tokens(name, seed, seqlen, tokenizer):
    if name == "wikitext2":
        return get_wikitext2(seed, seqlen, tokenizer)
    elif name == "c4":
        return get_c4(seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset {name}")


@torch.no_grad()
def evaluate_ppl(model, tokenizer, dataset_name, seed, seqlen, device="cuda"):
    input_ids = get_test_tokens(dataset_name, seed, seqlen, tokenizer)
    nsamples = input_ids.numel() // seqlen
    if nsamples == 0:
        raise ValueError(f"Too few tokens in {dataset_name} for seqlen={seqlen}")

    input_ids = input_ids[:, : (seqlen * nsamples)].view(nsamples, seqlen).to(device)

    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0

    progress = tqdm(range(nsamples), desc=f"Evaluating {dataset_name}")
    model.eval()

    for ii in progress:
        input = input_ids[ii, :].unsqueeze(0)  # (1, seqlen)
        output = model(
            input, use_cache=False, output_hidden_states=False, output_attentions=False
        )
        shift_logits = output.logits[:, :-1, :].contiguous()  # (1, seqlen-1, vocab)
        shift_labels = input[:, 1:].contiguous()  # (1, seqlen-1)

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        total_nll += loss.item()
        total_tokens += shift_labels.numel()

        current_ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
        progress.set_postfix(ppl=f"{current_ppl:.2f}")

    avg_nll_per_token = total_nll / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll_per_token)).item()
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model path",
        default="Qwen/Qwen3-8B-Base",
    )
    parser.add_argument("--seed", default=0, type=int)

    # evaluation seqlen
    parser.add_argument("--seqlen", default=8192, type=int)

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Loading model and tokenizer...")
    model = load_model(args.model, device_map=device, dtype=dtype)
    tokenizer = load_tokenizer(args.model)
    model.eval()

    if "base" in args.model.lower():
        datasets = ["wikitext2", "c4"]
        for dataset in datasets:
            ppl = evaluate_ppl(
                model, tokenizer, dataset, args.seed, args.seqlen, device=device
            )
            print(f"{dataset} perplexity: {ppl:.4f}")
    else:
        print("Running lm_eval...")
        lm_eval_cmd = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={args.model},enable_thinking=False,dtype=bfloat16",
            "--tasks",
            "arc_challenge,arc_easy,boolq,hellaswag",
            "--batch_size",
            "32",
        ]
        subprocess.run(lm_eval_cmd, check=True)


if __name__ == "__main__":
    main()