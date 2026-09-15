import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, BitsAndBytesConfig
#from peft import prepare_model_for_kbit_training
from torch.optim import AdamW
import bitsandbytes as bnb
from tqdm import tqdm
#from peft import prepare_model_for_kbit_training
import wandb
import time
import json
import shutil


class FakeQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super().__init__(in_features, out_features, bias)
        self.num_bits = num_bits

    def fake_quantize(self, weight: torch.Tensor) -> torch.Tensor: #changed for oprimisation, allocates only 2 tensors
        
        n_levels = 2 ** (self.num_bits -1) - 1
        clipping_ratio = 0.990

        with torch.no_grad():
            n_cols = weight.shape[1]
            clip_idx = min(int(clipping_ratio * n_cols), n_cols -1)
            threshold = weight.detach().abs().sort(dim=1).values[:, clip_idx].unsqueeze(1)
            threshold = threshold.clamp(min=1e-8)
            scale = threshold.div(n_levels).clamp(min=1e-8)

        q= weight.div(scale)
        q.round_()
        q.clamp_(-n_levels, n_levels)
        q.mul_(scale)
         
        return weight + (q-weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fake = self.fake_quantize(self.weight)
        

        return F.linear(x, w_fake, self.bias)


def replace_with_fake_quant_linear(model: nn.Module, num_bits: int = 8):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):

            if name == "lm_head":
                continue
            new_layer = FakeQuantLinear(
                module.in_features,
                module.out_features,
                bias = module.bias is not None,
                num_bits = num_bits,
            )

            with torch.no_grad():
                new_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias)

            new_layer = new_layer.to(
                device= module.weight.device,
                dtype= module.weight.dtype
            )
            setattr(model, name, new_layer)
        else:
            replace_with_fake_quant_linear(module, num_bits)

    return model

def save_checkpoint(step, student, optimizer, scheduler, all_indices, ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, f"step_{step:05d}")
    os.makedirs(ckpt_path, exist_ok= True)
    student.save_pretrained(ckpt_path)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

    training_state={
        "step": step,
        "all_indices": all_indices,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state().tolist(),
        "rng_cuda": torch.cuda.get_rng_state().tolist(),
    }

    with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
        json.dump({
            "step": step,
            "all_indices": all_indices,
            "rng_python": list(training_state["rng_python"][1]),
        },f)

    torch.save({
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python_state": random.getstate(),
    }, os.path.join(ckpt_path, "rng_states.pt"))
    print(f"-> checkpoint saved: {ckpt_path}")

def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    ckpt_folders= [
        f for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, f))
    ]
    if not ckpt_folders:
        return None

    ckpt_folders.sort(key= lambda x : int(x.replace("step_","")))
    return os.path.join(ckpt_dir, ckpt_folders[-1])

def load_checkpoint(ckpt_path, student_path, optimizer, scheduler, device):
    print(f" resuming from checkpoint: {ckpt_path}")

    student = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype= torch.bfloat16,
        trust_remote_code = True,
    )
    #print("before:",student.model.layers[0].self_attn.q_proj.weight.dtype)
    #student = replace_with_fake_quant_linear(student, num_bits=8)
    #print("after:",student.model.layers[0].self_attn.q_proj.weight.dtype)
    student.gradient_checkpointing_enable()
    student.train()
    student = student.to(device)
    '''n_linear = sum(
        isinstance(m, FakeQuantLinear)
        for m in student.modules()
    )'''
    #print(f"Replaced {n_linear} Linear layers with FakeQuantLinear.")



    opt_state = torch.load(
        os.path.join(ckpt_path, "optimizer.pt"),
        map_location = device
    )

    optimizer.load_state_dict(opt_state)

    sch_state = torch.load(os.path.join(ckpt_path, "scheduler.pt"))
    scheduler.load_state_dict(sch_state)

    with open(os.path.join(ckpt_path, "training_state.json")) as f:
        state = json.load(f)
    step = state["step"]
    all_indices = state["all_indices"]

    rng= torch.load(os.path.join(ckpt_path,"rng_states.pt"))
    torch.set_rng_state(rng["rng_torch"])
    torch.cuda.set_rng_state(rng["rng_cuda"])
    np.random.set_state(rng["rng_numpy"])
    random.setstate(rng["rng_python_state"])

    print(f"resumed from step {step}")
    return student, optimizer, scheduler, step, all_indices


def cleanup_old_checkpoints(ckpt_dir, keep_last_n=2):
    ckpt_folders = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, f))
    ], key = lambda x: int(x.replace("step_","")))
    to_delete = ckpt_folders[:-keep_last_n]

    for folder in to_delete:
        folder_path = os.path.join(ckpt_dir, folder)
        shutil.rmtree(folder_path)
        print(f"-> deleted old checkpoint: {folder_path}")


Student_path = "./StudentModel/deepseek-coder-1.3b/"
Cache_dir = "./poc_cache_deepseek_codeparrot_6.7B"
Ckpt_dir = "./poc_checkpoints_1.3B_KD_deepseek_6.7B"
os.makedirs(Ckpt_dir, exist_ok = True)


Num_epochs = 1
Max_steps = 30538
Warmup_steps= int(Max_steps * 0.05)
Batch_size = 2
LR = 3e-5
Temp = 4.0
Alpha = 0.7
Top_k = 50
Max_seq_len = 4096
Log_every = 50
Save_every = 1000
Device = "cuda:0"
start_time = time.time()


                   
print("scanning cache files")
all_indices = []
for fname in sorted(os.listdir(Cache_dir)):
    if not fname.endswith("_vals.npy"):
        continue
    idx = int(fname.replace("sample_","").replace("_vals.npy",""))
    vals= np.load(f"{Cache_dir}/sample_{idx:07d}_vals.npy")
    if vals.size > 0:
        all_indices.append(idx)
print(f"{len(all_indices)} valid samples")


latest_ckpt = find_latest_checkpoint(Ckpt_dir)

if latest_ckpt is not None:
    print("Starting from a found checkpoint")
    temp = AutoModelForCausalLM.from_pretrained(
        Student_path,
        dtype = torch.bfloat16,
        trust_remote_code= True,
        #device_map="auto",
    )

    #temp = replace_with_fake_quant_linear(student, num_bits =8)    
    temp= temp.to(Device)
    temp.train()

    optimizer = AdamW(temp.parameters(), lr= LR, weight_decay= 0.01)
    scheduler = get_scheduler(
        "cosine",
         optimizer = optimizer,
         num_warmup_steps = Warmup_steps,
         num_training_steps = Max_steps,
    )
    del temp
    torch.cuda.empty_cache()

    student, optimizer, scheduler, start_step, all_indices = load_checkpoint(
        latest_ckpt, Student_path, optimizer, scheduler, Device
    )


else:
    print("No checkpoint found, starting fresh")
    student = AutoModelForCausalLM.from_pretrained(
        Student_path,
        dtype = torch.bfloat16,
        trust_remote_code= True,
    )
    student.gradient_checkpointing_enable()
    student.config.use_cache = False

    #print("before:",student.model.layers[0].self_attn.q_proj.weight.dtype)
    #student = replace_with_fake_quant_linear(student, num_bits =8)
    #print("after:",student.model.layers[0].self_attn.q_proj.weight.dtype)

    '''n_linear = sum(
        isinstance(m, FakeQuantLinear)
        for m in student.modules()
    )'''
    student= student.to(Device)
    #print(f"Replaced {n_linear} Linear layers with FakeQuantLinear.")


    student.train()

    optimizer = AdamW(student.parameters(), lr= LR, weight_decay= 0.01)
    scheduler = get_scheduler(
        "cosine",
         optimizer = optimizer,
         num_warmup_steps = Warmup_steps,
         num_training_steps = Max_steps,
    )

    start_step = 0 
    random.shuffle(all_indices)
    
wandb.init(project= "qad-poc", config={
    "max_steps": Max_steps,
    "batch_size": Batch_size,
    "lr": LR,
    "temperature": Temp,
    "alpha": Alpha,
    "top_k": Top_k,
    "quantization": "fake_int8_symmetric_per_channel_ste",
    })


#Training 

step = start_step
running_loss= 0
running_kd = 0
running_ce = 0
total_kd = 0
ema_kd = None
skipped_batch = 0

while step < Max_steps:

    iter_start = time.time()
    batch_start = (step* Batch_size) % len(all_indices)
    batch_idxs = all_indices[batch_start: batch_start + Batch_size]
    batch_tokens =[]
    batch_vals = []
    batch_idxs_t = []


    for i in batch_idxs:
        tokens = np.load(f"{Cache_dir}/sample_{i:07d}_tokens.npy")[0]
        vals = np.load(f"{Cache_dir}/sample_{i:07d}_vals.npy")
        idxs = np.load(f"{Cache_dir}/sample_{i:07d}_idxs.npy")

        batch_tokens.append(tokens)
        batch_vals.append(vals)
        batch_idxs_t.append(idxs)

    max_len = max(t.shape[0] for t in batch_tokens)

    padded_tokens=[]
    padded_vals = []
    padded_idxs = []
    attn_masks = []

    for tokens, vals, idxs in zip(batch_tokens, batch_vals, batch_idxs_t):
        seq_len = tokens.shape[0]
        pad_len = max_len - seq_len


        padded_tokens.append(
            np.pad(tokens, (0, pad_len), constant_values=0)
        )

        padded_vals.append(
            np.pad(vals, ((0, pad_len), (0,0)), constant_values = 0.0)
        )

        padded_idxs.append(
            np.pad(idxs, ((0, pad_len), (0,0)), constant_values= 0)
        )

        mask = np.zeros(max_len, dtype= np.int64)
        mask[:seq_len]=1
        attn_masks.append(mask)

        
    input_ids = torch.tensor(np.stack(padded_tokens), dtype= torch.long).to(Device)
    teacher_vals = torch.tensor(np.stack(padded_vals), dtype= torch.float32).to(Device)
    teacher_idxs = torch.tensor(np.stack(padded_idxs), dtype= torch.long).to(Device)
    attention_mask = torch.tensor(np.stack(attn_masks), dtype= torch.long).to(Device)

    output = student(input_ids = input_ids, attention_mask = attention_mask, use_cache =False)
    student_logits = output.logits

    B, S, V = student_logits.shape


    t_soft = F.softmax(teacher_vals[:,:-1,:]/Temp,dim=-1)
    s_logits_topk = student_logits[:,:-1,:].gather(dim =2, index = teacher_idxs[:,:-1,:])
    s_log_topk = F.log_softmax(s_logits_topk / Temp, dim=-1)


    labels = input_ids[:,1:].clone()
    padding_mask = (attention_mask[:, 1:] == 0)
    labels = labels.masked_fill(padding_mask, -100)

    valid_tokens = (labels != -100).sum().item()
    if valid_tokens == 0:
        skipped_batch +=1
        optimizer.zero_grad(set_to_none= True)
        continue


    #kd loss
    kd_mask = attention_mask[:, :-1].float()
    kd_per_pos = F.kl_div(s_log_topk, 
                              t_soft, 
                              reduction = "none",
                              log_target = False,).sum(dim=-1)
        
    kd_loss = (kd_per_pos * kd_mask).sum() / kd_mask.sum()
    kd_loss = kd_loss * (Temp**2)
    if ema_kd is None:
        ema_kd = kd_loss.item()
    else:
        ema_kd = 0.99 * ema_kd + kd_loss.item() * 0.01
        
    #ce loss
    logits_shifted= student_logits[:,:-1,:].contiguous()
    if not torch.isfinite(logits_shifted).all():
        print("NaN/Inf in logits!")
        print("min:", logits_shifted.nan_to_num().min())
        print("max:", logits_shifted.nan_to_num().max())
    
    ce_loss = F.cross_entropy(
        logits_shifted.view(-1,V),
        labels.view(-1),
        ignore_index= -100,
    )

    loss = Alpha * kd_loss + (1-Alpha) * ce_loss
    

    '''if not torch.isfinite(loss):
        print(f"Step {step}")
        print("KD:", kd_loss.item())
        print("CE:", ce_loss.item())
        print("Student logits finite:", torch.isfinite(student_logits).all().item())
        print("Labels finite:", torch.isfinite(labels.float()).all().item())
        #print("Label min/max:", labels.min().item(), labels.max().item())
        print("Attention mask sum:", attention_mask.sum().item())
        optimizer.zero_grad(set_to_none=True)
        continue'''


    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    #gradient_clipping 
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()


    step +=1
    running_loss += loss.item()
    running_kd += kd_loss.item()
    running_ce +=ce_loss.item()
    total_kd += kd_loss.item()
    c_kd = total_kd/ step

    iter_time = time.time() - iter_start
    #logging
    if step % Log_every == 0:
        avg_loss = running_loss / Log_every
        avg_kd   = running_kd   / Log_every
        avg_ce   = running_ce   / Log_every

        print(f"step {step:>4}/ {Max_steps}  "
              f"loss={avg_loss:.4f}  "
              f"kd={avg_kd:.4f}  "
              f"c_kd = {c_kd:.4f} "
              f"ema_kd = {ema_kd:.4f} "
              f"ce={avg_ce:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}"
              
        )

        wandb.log({
            "loss":    avg_loss,
            "kd_loss": avg_kd,
            "c_kd": c_kd,
            "ema_kd": ema_kd,
            "ce_loss": avg_ce,
            "lr":      scheduler.get_last_lr()[0],
            "step":    step,
        })

        running_loss = running_kd = running_ce = 0.0


     #checkpoint
    if step % Save_every == 0:
        save_checkpoint(
            step, student, optimizer, scheduler, all_indices, Ckpt_dir
        )
        cleanup_old_checkpoints(Ckpt_dir, keep_last_n=2)


#save
final_path = f"{Ckpt_dir}/final"
student.save_pretrained(final_path)
print(f"\nTraining complete. Final model saved to {final_path}")

wandb.finish()

elapsed = time.time()- start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)
print(f"Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}")