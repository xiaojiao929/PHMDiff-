# The original code is licensed under MIT License, which is can be found at licenses/LICENSE_UVIT.txt.

import argparse
import os.path
from copy import deepcopy
from time import time
from omegaconf import OmegaConf

import apex
import torch
import accelerate

from torch.utils.data import DataLoader

from fid import calc
from models.PHMDiff import Precond_models
from train_utils.loss import Losses
from train_utils.datasets import MedicalImagingDataset 

from train_utils.helper import get_mask_ratio_fn, requires_grad, update_ema, unwrap_model

from sample import generate_with_net
from utils import dist, mprint, get_latest_ckpt, Logger, sample, \
    str2bool, parse_str_none, parse_int_list, parse_float_none


# ------------------------------------------------------------


def train_loop(args):
    # load configuration
    config = OmegaConf.load(args.config)
    
    if not args.no_amp:
        config.train.amp = 'fp16'
    else:
        config.train.amp = 'no'
    
    if config.train.tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    
    accelerator = accelerate.Accelerator(mixed_precision=config.train.amp, 
                                         gradient_accumulation_steps=config.train.grad_accum, 
                                         log_with='wandb')
    # setup wandb 
    if args.use_wandb:
        wandb_init_kwargs = {
            'entity': config.wandb.entity,
            'project': config.wandb.project,
            'group': config.wandb.group,
        }
        accelerator.init_trackers(config.wandb.project, config=OmegaConf.to_container(config), init_kwargs=wandb_init_kwargs)

    mprint('start training...')
    size = accelerator.num_processes
    rank = accelerator.process_index

    print(f'global_rank: {rank}, global_size: {size}')
    device = accelerator.device

    seed = args.global_seed 
    torch.manual_seed(seed)

    mprint(f"enable_amp: {not args.no_amp}, TF32: {config.train.tf32}")
    # Select batch size per GPU
    num_accumulation_rounds = config.train.grad_accum
    micro_batch = config.train.batchsize
    batch_gpu_total = micro_batch * num_accumulation_rounds
    global_batch_size = batch_gpu_total * size
    mprint(f"Global batchsize: {global_batch_size},  batchsize per GPU: {batch_gpu_total}, micro_batch: {micro_batch}.")

    class_dropout_prob = config.model.class_dropout_prob
    log_every = config.log.log_every
    ckpt_every = config.log.ckpt_every
    
    mask_ratio_fn = get_mask_ratio_fn(config.model.mask_ratio_fn, config.model.mask_ratio, config.model.mask_ratio_min)

    # Setup an experiment folder
    model_name = config.model.model_type.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    data_name = config.data.dataset
    if args.ckpt_path is not None and args.use_ckpt_path:  # use the existing exp path (mainly used for fine-tuning)
        checkpoint_dir = os.path.dirname(args.ckpt_path)
        experiment_dir = os.path.dirname(checkpoint_dir)
        exp_name = os.path.basename(experiment_dir)
    else:  # start a new exp path (and resume from the latest checkpoint if possible)
        cond_gen = 'cond' if config.model.num_classes else 'uncond'
        exp_name = f'{model_name}-{config.model.precond}-{data_name}-{cond_gen}-m{config.model.mask_ratio}-de{int(config.model.use_decoder)}' \
                   f'-mae{config.model.mae_loss_coef}-bs-{global_batch_size}-lr{config.train.lr}{config.log.tag}'
        experiment_dir = f"{args.results_dir}/{exp_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        if args.ckpt_path is None:
            args.ckpt_path = get_latest_ckpt(checkpoint_dir)  # Resumes from the latest checkpoint if it exists
    mprint(f"Experiment directory created at {experiment_dir}")

    if accelerator.is_main_process:
        logger = Logger(file_name=f'{experiment_dir}/log.txt', file_mode="a+", should_flush=True)
    
    mprint(f"Experiment directory created at {experiment_dir}")
    # Setup dataset
    dataset = ImageNetLatentDataset(
        config.data.root, resolution=config.data.resolution, 
        num_channels=config.data.num_channels, xflip=config.train.xflip, 
        feat_path=config.data.feat_path, feat_dim=config.model.ext_feature_dim)

    loader = DataLoader(
        dataset, batch_size=batch_gpu_total, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True,
        drop_last=True
    )
    mprint(f"Dataset contains {len(dataset):,} images ({config.data.root})")

    steps_per_epoch = len(dataset) // global_batch_size
    mprint(f"{steps_per_epoch} steps per epoch")

    model = Precond_models[config.model.precond](
        img_resolution=config.model.in_size,
        img_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        model_type=config.model.model_type,
        use_decoder=config.model.use_decoder,
        mae_loss_coef=config.model.mae_loss_coef,
        pad_cls_token=config.model.pad_cls_token
    ).to(device)

    # Note that parameter initialization is done within the model constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    mprint(f"{config.model.model_type} ((use_decoder: {config.model.use_decoder})) Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    mprint(f'extras: {model.model.extras}, cls_token: {model.model.cls_token}')

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=config.train.lr, adam_w_mode=True, weight_decay=0)

    # Load checkpoints
    train_steps_start = 0
    epoch_start = 0

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=args.use_strict_load)
        ema.load_state_dict(ckpt['ema'], strict=args.use_strict_load)
        mprint(f'Load weights from {args.ckpt_path}')
        if args.use_strict_load:
            optimizer.load_state_dict(ckpt['opt'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        mprint(f'Load optimizer state..')
        train_steps_start = int(os.path.basename(args.ckpt_path).split('.pt')[0])
        epoch_start = train_steps_start // steps_per_epoch
        mprint(f"train_steps_start: {train_steps_start}")
        del ckpt # conserve memory

        # FID evaluation for the loaded weights
        if args.enable_eval:
            start_time = time()
            args.outdir = os.path.join(experiment_dir, 'fid', f'edm-steps{args.num_steps}-ckpt{train_steps_start}_cfg{args.cfg_scale}')
            os.makedirs(args.outdir, exist_ok=True)
            generate_with_net(args, ema, device)
            dist.barrier()
            fid = calc(args.outdir, config.eval.ref_path, args.num_expected, args.global_seed, args.fid_batch_size)
            mprint(f"time for fid calc: {time() - start_time}")
            if args.use_wandb:
                accelerator.log({f'eval/fid': fid}, step=train_steps_start)
            mprint(f'guidance: {args.cfg_scale} FID: {fid}')
            dist.barrier()

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model = torch.compile(model)

    # Setup loss
    loss_fn = Losses[config.model.precond]()

    # Prepare models for training:
    if args.ckpt_path is None:
        assert train_steps_start == 0
        raw_model = unwrap_model(model)
        update_ema(ema, raw_model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = train_steps_start
    log_steps = 0
    running_loss = 0
    start_time = time()
    mprint(f"Training for {config.train.epochs} epochs...")
    for epoch in range(epoch_start, config.train.epochs):
        mprint(f"Beginning epoch {epoch}...")
        for x, cond in loader:
            x = x.to(device)
            y = cond.to(device)
            x = sample(x)
            # Accumulate gradients.
            loss_batch = 0
            model.zero_grad(set_to_none=True)
            curr_mask_ratio = mask_ratio_fn((train_steps - train_steps_start) / config.train.max_num_steps)
            if class_dropout_prob > 0:
                y = y * (torch.rand([y.shape[0], 1], device=device) >= class_dropout_prob)

            for round_idx in range(num_accumulation_rounds):
                x_ = x[round_idx * micro_batch: (round_idx + 1) * micro_batch]
                y_ = y[round_idx * micro_batch: (round_idx + 1) * micro_batch]

                with accelerator.accumulate(model):
                    loss = loss_fn(net=model, images=x_, labels=y_, 
                                    mask_ratio=curr_mask_ratio,
                                    mae_loss_coef=config.model.mae_loss_coef)
                    loss_mean = loss.mean()
                    accelerator.backward(loss_mean)

                    # Update weights with lr warmup.
                    lr_cur = config.train.lr * min(train_steps * global_batch_size / max(config.train.lr_rampup_kimg * 1000, 1e-8), 1)
                    for g in optimizer.param_groups:
                        g['lr'] = lr_cur
                    optimizer.step()
                    loss_batch += loss_mean.item()

            raw_model = unwrap_model(model)
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss_batch
            log_steps += 1
            train_steps += 1
            if train_steps > (train_steps_start + config.train.max_num_steps):
                break
            if train_steps % log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / size
                mprint(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                mprint(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')
                mprint(f'Reserved GPU memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB')

                if args.use_wandb:
                    accelerator.log({f'train/loss': avg_loss, 'train/lr': lr_cur}, step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % ckpt_every == 0 and train_steps > train_steps_start:
                if rank == 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    mprint(f"Saved checkpoint to {checkpoint_path}")
                    del checkpoint  # conserve memory
                dist.barrier()

                # FID evaluation during training
                if args.enable_eval:
                    start_time = time()
                    args.outdir = os.path.join(experiment_dir, 'fid', f'edm-steps{args.num_steps}-ckpt{train_steps}_cfg{args.cfg_scale}')
                    os.makedirs(args.outdir, exist_ok=True)
                    generate_with_net(args, ema, device, rank, size)

                    dist.barrier()
                    fid = calc(args.outdir, args.ref_path, args.num_expected, args.global_seed, args.fid_batch_size)
                    mprint(f"time for fid calc: {time() - start_time}, fid: {fid}")
                    if args.use_wandb:
                        accelerator.log({f'eval/fid': fid}, step=train_steps)
                    mprint(f'Guidance: {args.cfg_scale}, FID: {fid}')
                    dist.barrier()
                start_time = time()
                
    if accelerator.is_main_process:
        logger.close()
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training parameters')
    # basic config
    parser.add_argument('--config', type=str, required=True, help='path to config file')

    # training
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt_path", type=parse_str_none, default=None)

    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--no_amp', action='store_true', help="Disable automatic mixed precision.")

    parser.add_argument("--use_wandb", action='store_true', help='enable wandb logging')
    parser.add_argument("--use_ckpt_path", type=str2bool, default=True)
    parser.add_argument("--use_strict_load", type=str2bool, default=True)
    parser.add_argument("--tag", type=str, default='')

    # sampling
    parser.add_argument('--enable_eval', action='store_true', help='enable fid calc during training')
    parser.add_argument('--seeds', type=parse_int_list, default='0-49999', help='Random seeds (e.g. 1,2,5-10)')
    parser.add_argument('--subdirs', action='store_true', help='Create subdirectory for every 1000 seeds')
    parser.add_argument('--class_idx', type=int, default=None, help='Class label  [default: random]')
    parser.add_argument('--max_batch_size', type=int, default=50, help='Maximum batch size per GPU during sampling, must be a factor of 50k if torch.compile is used')

    parser.add_argument("--cfg_scale", type=parse_float_none, default=None, help='None = no guidance, by default = 4.0')

    parser.add_argument('--num_steps', type=int, default=40, help='Number of sampling steps')
    parser.add_argument('--S_churn', type=int, default=0, help='Stochasticity strength')
    parser.add_argument('--solver', type=str, default=None, choices=['euler', 'heun'], help='Ablate ODE solver')
    parser.add_argument('--discretization', type=str, default=None, choices=['vp', 've', 'iddpm', 'edm'], help='Ablate ODE solver')
    parser.add_argument('--schedule', type=str, default=None, choices=['vp', 've', 'linear'], help='Ablate noise schedule sigma(t)')
    parser.add_argument('--scaling', type=str, default=None, choices=['vp', 'none'], help='Ablate signal scaling s(t)')
    parser.add_argument('--pretrained_path', type=str, default='assets/stable_diffusion/autoencoder_kl.pth', help='Autoencoder ckpt')

    parser.add_argument('--ref_path', type=str, default='assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz', help='Dataset reference statistics')
    parser.add_argument('--num_expected', type=int, default=50000, help='Number of images to use')
    parser.add_argument('--fid_batch_size', type=int, default=64, help='Maximum batch size per GPU')

    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    train_loop(args)
