import torch
import time
from dataloader import DataLoaderLite
from gpt2 import *
from modules import *
import math
from parser import *
import subprocess
from seed import *
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os


def main():
    # setting seed
    set_seed()

    # ----------------------------------------------------------------------------------------------------------
    # simple launch: python train_gpt2.py
    # DDP launch for eg 8 GPUS: torchrun --standlone --nproc_per_node=8 train_gpt2.py

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "CUDA REQUIRED FOR THIS DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # ----------------------------------------------------------------------------------------------------------

    # load the data with gradient accumulation for simulating larger batch size. batch_size = 524288 = 2**19 ~0.5M 
    total_batch_size = 524288 
    B = 4 # micro-batch size
    T = 1024 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process: # to avoid printing this multiple times in DDP for each GPU.
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    trainloader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, master_process=master_process, split='train')

    # parsing the arguments
    parser = parsing()

    # Model Initialization: to use fp16 use .half() and to use fp32 use .float()
    model = get_model(parser) 
    model.to(device)
    # model = torch.compile(model) did not work for since my cuda compatability is less than 7. 
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the raw unwrapped model



    # setting the learning cosine decay rate schedular 
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073
    def get_lr(it):
        # 1) linear warmup for warmup iterations
        if it < warmup_steps:
            return max_lr * (it+1)/ warmup_steps

        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr

        # 3) in between, we use cosine decay down to min_lr
        decay_rate = (it-warmup_steps) / (max_steps - warmup_steps)   
        assert 0 <= decay_rate <= 1

        # coeff starts at 0 and goes to 0.
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate))
        return min_lr + coeff * (max_lr - min_lr)



    # optimizer: hyperparameters are replicated from the GPT-3 paper.
    # optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=36e-4, device_type=device)

   
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_acum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y  = trainloader.next_batch()
            x, y = x.to(device), y.to(device)
            y = y.type(torch.LongTensor)
            # this is automatic mixed precision (AMP) training that changes precision dynamically
            with torch.autocast(device_type=device, dtype=torch.float16): 
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradiant accumulation.
            # because the gradients just add up over the accumulation steps.
            # addition of gradients corresponds to a SUM in the objective, but 
            # instead of a sum we want MEAN. Scale the loss here so it comes out right.
            loss = loss / grad_accum_steps
            loss_acum += loss.detach()
            if ddp:
                model.require_backward_grad_sync =  (micro_step==grad_accum_steps-1) # only synchorize grads at the end of accumulation so each rank will have the avg of all the gradients.
            loss.backward()

        if ddp:
            dist.all_reduce(loss_acum, op=dist.ReduceOp.SUM)
        # gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping

        # determine and set the learning rate for this iteration.
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # wait for GPU to finish.
        torch.cuda.synchronize()

        # time difference in ms
        t1 = time.time()
        dt = (t1-t0)
        tokens_processed = trainloader.B * trainloader.T * grad_accum_steps * ddp_world_size
        tokens_per_second = tokens_processed / dt 
        if master_process:
            print(f"step {step} | loss: {loss_acum.item():.6f} | norm: {norm:.4f}  | lr: {lr:.4e}| time dt: {dt*1000} ms | tok/sec: {tokens_per_second}")
    
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    print("Downloading the fineweb educational 10B token dataset...\n")
    subprocess.run(["python", "fineweb.py"])
    print("Dataset downloaded successfully!\n")
    main()
