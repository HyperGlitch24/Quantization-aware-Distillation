import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from datasets import load_dataset
from tqdm import tqdm 
from pathlib import Path
from datasets import load_from_disk

#config
Teacher_path = './StudentModel/deepseek-coder-6.7b/'
Cache_dir = "./poc_cache_deepseek_codeparrot_6.7B"
Max_seq_len = 4096
top_k = 50
batch_size = 4
os.makedirs(Cache_dir, exist_ok= True)

print(f"teacher path exists? {os.path.exists(Teacher_path)}")

config = AutoConfig.from_pretrained(Teacher_path, local_files_only = True)
print(f"teacher dtype: {config.dtype}")

#Load Teacher and setup for inference

teacher = AutoModelForCausalLM.from_pretrained(
    Teacher_path,
    dtype = torch.bfloat16, 
    device_map = {"":"cuda:7"},
    trust_remote_code= True,
    local_files_only= True,
)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

#Tokenizer

tokenizer = AutoTokenizer.from_pretrained(Teacher_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#load dataset
print("Loading code parrot clean datset from disk...")
ds = load_from_disk("./Code_parrot_dataset")

print(ds)

print(f"found {len(ds)} samples")

#caching loop
existing = set()
for fname in os.listdir(Cache_dir):
    if fname.startswith("sample_") and fname.endswith("_vals.npy"):
        idx = int(fname.replace("sample_", "").replace("_vals.npy",""))
        existing.add(idx)

print(f"already cached: {len(existing)} samples. resuming")
                  
#caching

with torch.no_grad():
    for idx, example in enumerate(tqdm(ds, desc="Caching teacher logits")):


        if idx in existing:
            continue

        code = example.get("code","").strip()
        
        if not code:
            print(f"Warning: empty code at {idx} skipping")
            continue

        tokens = tokenizer(
            code,
            truncation = True,
            max_length= Max_seq_len, 
            return_tensors="pt"
            )
        input_ids = tokens["input_ids"].to("cuda:7")
        attention_mask = tokens["attention_mask"].to("cuda:7")

        outputs = teacher(input_ids = input_ids, attention_mask = attention_mask)
        logits = outputs.logits[0]


        top_vals, top_idxs = torch.topk(logits, k=top_k, dim=-1)

        np.save(f"{Cache_dir}/sample_{idx:07d}_vals.npy", top_vals.cpu().float().numpy())
        np.save(f"{Cache_dir}/sample_{idx:07d}_idxs.npy", top_idxs.cpu().to(torch.int32).numpy())
        np.save(f"{Cache_dir}/sample_{idx:07d}_tokens.npy", input_ids.cpu().to(torch.int32).numpy())


        del outputs, logits,top_vals, top_idxs
        if idx % 100 == 0:
            torch.cuda.empty_cache()

        

print(f"Caching complete in {Cache_dir}")