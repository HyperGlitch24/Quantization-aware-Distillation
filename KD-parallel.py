import os
os.environ["DS_BUILD_FUSED_ADAM"] = "0"
os.environ["DS_BUILD_FUSED_LAMB"] = "0"
os.environ["DS_BUILD_UTILS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from safetensors.torch import load_file
import glob
from transformers import AutoConfig
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig  
from torch.optim import AdamW
import bitsandbytes as bnb
from tqdm import tqdm
import wandb
import time
import shutil
import json
import deepspeed
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.accelerator import get_accelerator

'''accelerator = Accelerator(
    mixed_precision = "bf16"
)
accelerator.state.deepspeed_plugin.deepspeed_config[
    "train_micro_batch_size_per_gpu"
] = Batch_size
device = accelerator.device'''

class FakeQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super().__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.clipping_ratio = 0.999
        # No dtype casting here — handled by weight copy in replace()

    def fake_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        n_levels = 2 ** (self.num_bits - 1) - 1  # 7 for INT4
        with torch.no_grad():
            # MinMax — no clipping, paper validated this
            max_val = weight.detach().abs().amax(dim=1, keepdim=True)
            scale = (max_val / n_levels).clamp(min=1e-8)

        q = weight.div(scale)
        q.round_()
        q.clamp_(-n_levels, n_levels)
        q.mul_(scale)

        return weight + (q - weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fake = self.fake_quantize(self.weight)
        return F.linear(x, w_fake, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No GatheredParameters — ZeRO-3 handles allgather automatically
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
                bias=module.bias is not None,
                num_bits=num_bits,
            )

            # Weights are ZeRO-3 sharded — must gather to copy
            with deepspeed.zero.GatheredParameters(
                list(module.parameters()),
                modifier_rank=0
            ):
                with torch.no_grad():
                    new_layer.weight = nn.Parameter(
                        module.weight.data.clone().to(torch.bfloat16)
                    )
                    if module.bias is not None:
                        new_layer.bias = nn.Parameter(
                            module.bias.data.clone().to(torch.bfloat16)
                        )

            setattr(model, name, new_layer)
        else:
            replace_with_fake_quant_linear(module, num_bits)

    return model

def save_checkpoint(step, engine, scheduler, all_indices, ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, f"step_{step:05d}")
    os.makedirs(ckpt_path, exist_ok=True)

    # DeepSpeed saves model weights + optimizer states across all ranks
    engine.save_checkpoint(ckpt_dir, tag=f"step_{step:05d}")
    #                      ↑ ckpt_dir not ckpt_path — DeepSpeed creates
    #                        the tag subfolder itself inside ckpt_dir

    if is_main_process:
        torch.save(
            scheduler.state_dict(),
            os.path.join(ckpt_path, "scheduler.pt")
        )

        with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
            json.dump({
                "step": step,
                "all_indices": all_indices,
            }, f)

        torch.save({
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python_state": random.getstate(),
            **{f"rng_cuda_rank_{r}": torch.cuda.get_rng_state(r) 
               for r in range(torch.cuda.device_count())},
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

def load_checkpoint(ckpt_path, engine, scheduler):
    print(f"Resuming from checkpoint: {ckpt_path}")

    # DeepSpeed loads all shards correctly across all ranks
    _, client_state = engine.load_checkpoint(
        ckpt_path,
        tag=os.path.basename(ckpt_path),
        load_optimizer_states=True,
        load_lr_scheduler_states=False,  # we handle scheduler separately
    )

    # Load scheduler
    sch_state = torch.load(
        os.path.join(ckpt_path, "scheduler.pt"),
        map_location="cpu"
    )
    scheduler.load_state_dict(sch_state)

    # Load training state
    with open(os.path.join(ckpt_path, "training_state.json")) as f:
        state = json.load(f)
    step = state["step"]
    all_indices = state["all_indices"]

    # Load RNG states
    rng = torch.load(os.path.join(ckpt_path, "rng_states.pt"))
    torch.set_rng_state(rng["rng_torch"])
    torch.cuda.set_rng_state(rng[f"rng_cuda_rank_{local_rank}"])
    np.random.set_state(rng["rng_numpy"])
    random.setstate(rng["rng_python_state"])

    print(f"Resumed from step {step}")
    return step, all_indices



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
    ], key=lambda x: int(x.replace("step_", "")))
    to_delete = ckpt_folders[:-keep_last_n]
    for folder in to_delete:
        folder_path = os.path.join(ckpt_dir, folder)
        shutil.rmtree(folder_path)
        print(f"-> deleted old checkpoint: {folder_path}")




Student_path = "./StudentModel/Qwen2.5-Coder-7B/"
Cache_dir = "./poc_cache_Qwen_codeparrot"
Ckpt_dir = "./poc_checkpoints_7B_4bit_KD_Qwen_codeparrot"
os.makedirs(Ckpt_dir, exist_ok = True)
torch.set_default_device("cpu")

ds_config_zero3= "ds_config_zero3.json"
with open(ds_config_zero3) as f:
    ds_config = json.load(f)

local_rank = int(os.environ.get("LOCAL_RANK",0))
is_main_process = (local_rank == 0 )
deepspeed.init_distributed()  # ← initializes process group

Device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(Device)

print(f"[Rank {local_rank}] Using GPU: {torch.cuda.current_device()} "
      f"({torch.cuda.get_device_name(torch.cuda.current_device())})")

#dschf = HfDeepSpeedConfig(ds_config_zero3)

Num_epochs = 1
#Max_steps = 15269
Max_steps = 5089
Warmup_steps= int(Max_steps * 0.05)
#Batch_size = 4
Batch_size = 3
#LR = 8e-5
LR = 5e-5
Total_optimizer_steps = 637
Warmup_optimizer_steps = 32
Temp = 4.0
Alpha = 0.7
Top_k = 50
Max_seq_len = 4096
Log_every = 50
Save_every = 1000
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
    print(f"Resuming from checkpoint: {latest_ckpt}")

    # Same simple loading as fresh start
    # Rank 0 saves to temp, all ranks load
    TEMP_CKPT = "/tmp/student_resume_weights"

    if local_rank == 0:
        student_temp = AutoModelForCausalLM.from_pretrained(
            Student_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        os.makedirs(TEMP_CKPT, exist_ok=True)
        student_temp.save_pretrained(TEMP_CKPT, safe_serialization=True)
        student_temp.config.save_pretrained(TEMP_CKPT)
        del student_temp
        torch.cuda.empty_cache()

    dist.barrier(device_ids=[local_rank])

    student = AutoModelForCausalLM.from_pretrained(
        TEMP_CKPT,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    #student = replace_with_fake_quant_linear(student, num_bits=8)
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    student.config.use_cache = False

    optimizer = AdamW(student.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=Warmup_steps,
        num_training_steps=Max_steps,
    )

    student_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=student,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=student.parameters(),
        config=ds_config,
    )

    # DeepSpeed loads checkpoint weights + optimizer states
    # This overwrites the pretrained weights with checkpoint weights
    _, client_state = student_engine.load_checkpoint(
        os.path.dirname(latest_ckpt),
        tag=os.path.basename(latest_ckpt),
        load_optimizer_states=True,
        load_lr_scheduler_states=False,
    )

    # Load scheduler and RNG separately
    sch_state = torch.load(
        os.path.join(latest_ckpt, "scheduler.pt"),
        map_location="cpu"
    )
    scheduler.load_state_dict(sch_state)

    with open(os.path.join(latest_ckpt, "training_state.json")) as f:
        state = json.load(f)
    start_step = state["step"]
    all_indices = state["all_indices"]

    rng = torch.load(os.path.join(latest_ckpt, "rng_states.pt"))
    torch.set_rng_state(rng["rng_torch"])
    torch.cuda.set_rng_state(rng["rng_cuda"])
    np.random.set_state(rng["rng_numpy"])
    random.setstate(rng["rng_python_state"])

    dist.barrier(device_ids=[local_rank])
    if local_rank == 0:
        shutil.rmtree(TEMP_CKPT, ignore_errors=True)

    Device = student_engine.device
    print(f"Resumed from step {start_step}")


else:
    print("No checkpoint found, starting fresh")

    # Step 1: Create sharded EMPTY structure only
    # from_config — no weight loading at all
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        hf_config = AutoConfig.from_pretrained(
            Student_path,
            trust_remote_code=True
        )
        student = AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=torch.bfloat16,
        )

    # Verify: should be ~5GB (sharded empty shells) not 14GB
    print(f"Rank {local_rank}: GPU mem after structure creation: "
          f"{torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Step 2: Load weights from safetensors shards on CPU
    shard_files = sorted(glob.glob(f"{Student_path}/*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors found in {Student_path}")

    if is_main_process:
        print(f"Found {len(shard_files)} safetensor shards, loading...")

    full_state_dict = {}
    for shard_file in shard_files:
        shard_dict = load_file(shard_file, device="cpu")
        full_state_dict.update(shard_dict)

    if is_main_process:
        print(f"State dict loaded: {len(full_state_dict)} tensors")
        print(f"Sample keys: {list(full_state_dict.keys())[:3]}")

    # Step 3: Fill sharded parameters with actual weights
    loaded = 0
    missing = []
    for name, param in student.named_parameters():
        if name in full_state_dict:
            with deepspeed.zero.GatheredParameters(
                param,
                modifier_rank=0
            ):
                if dist.get_rank() == 0:
                    param.data.copy_(
                        full_state_dict[name]
                        .to(dtype=torch.bfloat16)
                        .to(param.device)
                    )
            loaded += 1
        else:
            missing.append(name)

    if is_main_process:
        print(f"Loaded {loaded} parameters")
        if missing:
            print(f"Missing {len(missing)} parameters: {missing[:5]}")

    del full_state_dict
    torch.cuda.empty_cache()
    dist.barrier(device_ids=[local_rank])

    # Verify weights are now filled
    with deepspeed.zero.GatheredParameters(
        next(student.parameters()),
        modifier_rank=None
    ):
        first_p = next(student.parameters())
        print(f"Rank {local_rank}: "
              f"first param norm={first_p.norm().item():.4f} "
              f"(should be non-zero if weights loaded correctly)")

    # Step 4: Replace linear layers with FakeQuantLinear
    student = replace_with_fake_quant_linear(student, num_bits=4)

    if is_main_process:
        n_linear = sum(
            isinstance(m, FakeQuantLinear) for m in student.modules()
        )
        print(f"Replaced {n_linear} Linear layers with FakeQuantLinear.")

    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    student.config.use_cache = False

    
    optimizer = AdamW(student.parameters(), lr=LR, weight_decay=0.0)

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=32,
        num_training_steps=637,
    )
    student_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=student,
        optimizer= optimizer,
        lr_scheduler=scheduler,
        model_parameters=student.parameters(),
        config=ds_config_zero3,
    )

    


    Device = student_engine.device
    start_step = 0
    random.shuffle(all_indices)
    
if is_main_process:
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

    output = student_engine(input_ids = input_ids, attention_mask = attention_mask, use_cache =False)
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
    '''if not torch.isfinite(logits_shifted).all():
        print("NaN/Inf in logits!")
        print("min:", logits_shifted.nan_to_num().min())
        print("max:", logits_shifted.nan_to_num().max())'''
    
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


    '''optimizer.zero_grad(set_to_none=True)
    loss.backward()

    #gradient_clipping 
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()'''

    student_engine.backward(loss)
    student_engine.step()

    '''torch.cuda.empty_cache()
    torch.cuda.synchronize()'''
    get_accelerator().empty_cache()

    


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

        if is_main_process:
            current_lr = student_engine.get_lr()[0]
            print(f"step {step:>4}/ {Max_steps}  "
              f"loss={avg_loss:.4f}  "
              f"kd={avg_kd:.4f}  "
              f"c_kd = {c_kd:.4f} "
              f"ema_kd = {ema_kd:.4f} "
              f"ce={avg_ce:.4f}  "
              f"lr={current_lr:.2e}"
              f"skipped={skipped_batch}"
              f"iter_time={iter_time}"
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
    if step % Save_every == 0 and step > 0:
        save_checkpoint(step, student_engine, scheduler, all_indices, Ckpt_dir)
        dist.barrier(device_ids=[local_rank])  # wait for save to complete
        if is_main_process:
            cleanup_old_checkpoints(Ckpt_dir, keep_last_n=2)
        dist.barrier(device_ids=[local_rank])  # wait for cleanup




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
