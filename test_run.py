import argparse
import glob
import logging
import math
import os
import random
import time


import torch

from typing import Tuple
from torch import nn
from torch import Tensor as T

from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, BiEncoderNllLoss, BiEncoderBatch
from dpr.options import add_encoder_params, add_training_params, setup_args_gpu, set_seed, print_args, \
    get_encoder_params_state, add_tokenizer_params, set_encoder_params_from_state
from dpr.utils.data_utils import ShardedDataIterator, read_data_from_json_files, Tensorizer
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import setup_for_distributed_mode, move_to_device, get_schedule_linear, CheckpointState, \
    get_model_file, get_model_obj, load_states_from_checkpoint

from train_dense_encoder import train_dpr_parser, BiEncoderTrainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser = train_dpr_parser(parser)

cmd = '--encoder_model_type hf_bert '\ 
      '--pretrained_model_cfg bert-base-uncased '\
	  '--train_file /ssd3/DPR/data/retriever/trivia-dev.json '\
      '--dev_file  /ssd3/DPR/data/retriever/trivia-dev.json '\
	  '--output_dir /ssd3/DPR/output '\
      '--fp16'

args = parser.parse_args(cmd.split())


if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
        args.gradient_accumulation_steps))

if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)

setup_args_gpu(args)
set_seed(args)
print_args(args)

trainer = BiEncoderTrainer(args)

if args.train_file is not None:
    trainer.run_train()
elif args.model_file and args.dev_file:
    logger.info("No train files are specified. Run 2 types of validation for specified model file")
    trainer.validate_nll()
    trainer.validate_average_rank()
else:
    logger.warning("Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do.")

