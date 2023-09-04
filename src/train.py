import os, argparse

from easydict import EasyDict

from cvhelpers.misc import prepare_logger
from cvhelpers.torch_helpers import setup_seed

from data_loaders import get_dataloader
from models import get_model
from trainer import Trainer
from utils.misc import load_config
import torch

# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# def init_seeds(seed = 0, cuda_deterministic= True):
#     random.seed(seed)
#     np.random


#############
# Argparse. We use command line arguments for training options.
# Model and dataset options are stored in the .yaml config file
#############
parser = argparse.ArgumentParser()
# General
parser.add_argument('--config', type=str, help='Path to the config file.')
# Logging
parser.add_argument('--logdir', type=str, default='../logs',
                    help='Directory to store logs, summaries, checkpoints.')
parser.add_argument('--dev', action='store_true',
                    help='If true, will ignore logdir and log to ../logdev instead')
parser.add_argument('--name', type=str,
                    help='Experiment name (used to name output directory')
parser.add_argument('--summary_every', type=int, default=500,
                    help='Interval to save tensorboard summaries')
parser.add_argument('--validate_every', type=int, default=-1,
                    help='Validation interval. Default: every epoch')
parser.add_argument('--debug', action='store_true',
                    help='If set, will enable autograd anomaly detection')
# Misc
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of worker threads for dataloader')
# Training and model options
parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
parser.add_argument('--nb_sanity_val_steps', type=int, default=2,
                    help='Number of validation sanity steps to run before training.')

# DDP
parser.add_argument("--local_rank", default=-1, type=int)

opt = parser.parse_args()
local_rank = opt.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# Override config if --resume is passed
if opt.config is None:
    if opt.resume is None or not os.path.exists(opt.resume):
        print('--config needs to be supplied unless resuming from checkpoint')
        exit(-1)
    else:
        resume_folder = opt.resume if os.path.isdir(opt.resume) else os.path.dirname(opt.resume)
        opt.config = os.path.normpath(os.path.join(resume_folder, '../config.yaml'))
        if os.path.exists(opt.config):
            print(f'Using config file from checkpoint directory: {opt.config}')
        else:
            print('Config not found in resume directory')
            exit(-2)

cfg = EasyDict(load_config(opt.config))

# Hack: Stores different datasets to its own subdirectory
opt.logdir = os.path.join(opt.logdir, cfg.dataset)

if opt.name is None and len(cfg.get('expt_name', '')) > 0:
    opt.name = cfg.expt_name
logger, opt.log_path = prepare_logger(opt)

# Save config to log
config_out_fname = os.path.join(opt.log_path, 'config.yaml')
with open(opt.config, 'r') as in_fid, open(config_out_fname, 'w') as out_fid:
    out_fid.write(f'# Original file name: {opt.config}\n')
    out_fid.write(in_fid.read())


def main():
    rank = torch.distributed.get_rank()
    setup_seed(rank+1)
    train_loader = get_dataloader(cfg, phase='train', num_workers=opt.num_workers)
    if rank == 0:
        val_loader = get_dataloader(cfg, phase='val', num_workers=opt.num_workers)
    else:
        val_loader = None

    # torch.cuda.set_device(rank)
    # torch.cuda.empty_cache()


    Model = get_model(cfg.model)
    model = Model(cfg, local_rank=local_rank)
    trainer = Trainer(opt, niter=cfg.niter, grad_clip=cfg.grad_clip)
    trainer.fit(model, local_rank, train_loader, val_loader)


if __name__ == '__main__':
    main()
