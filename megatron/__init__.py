# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
from deepspeed.accelerator import get_accelerator

from .package_info import (
    __description__,
    __contact_names__,
    __url__,
    __download_url__,
    __keywords__,
    __license__,
    __package_name__,
    __version__,
)

from .global_vars import get_args
from .global_vars import get_current_global_batch_size
from .global_vars import get_num_microbatches
from .global_vars import update_num_microbatches
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .initialize  import initialize_megatron

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ
    
def is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
            is_aml() and torch.distributed.get_rank() % get_accelerator().device_count() == 0
            ):
            return True
        else:
            return False
    else:
        return True