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

"""Pretrain utilities."""

from datetime import datetime
import bisect
import math
import sys
import time
import json
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.checkpointing import get_checkpoint_groupid_filename
from megatron.model.module import Float16Module
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard, log_restart_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.utils import check_adlr_autoresume_termination, get_parameters_in_billions
from megatron.utils import unwrap_model, found_kill_switch
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_with_interleaving
from megatron.utils import report_memory, flops_calculator
from megatron.global_vars import codecarbon_tracker_start, codecarbon_tracker_stop
from megatron.data.dataset_utils import analyze_data_prefix
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import deepspeed
import os

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             forward_step_func,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()

    if found_kill_switch():
        print_datetime(f"Detected kill switch at {args.kill_switch_path}. Exiting")
        sys.exit()

    codecarbon_tracker_start()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    timers = get_timers()

    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, 'r', encoding='utf-8'))
        if "curriculum_learning" in args.deepspeed_configuration and \
            "enabled" in args.deepspeed_configuration["curriculum_learning"]:
            args.curriculum_learning = args.deepspeed_configuration[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning and \
            args.pipeline_model_parallel_size >= 1:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_configuration["curriculum_learning"])


    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    args.parameters_in_billions_no_embedding = get_parameters_in_billions(model, exclude_embeddings=True)
    print_rank_0(f'estimated model parameters: {get_parameters_in_billions(model)}')
    print_rank_0(f'estimated model parameters without embeddings: {get_parameters_in_billions(model, exclude_embeddings=True)}')
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid/test-data-iterators-setup').start()
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
        valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
        test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)

    if args.data_path is not None and len(args.data_path) > 1:
        prefixes, weights = analyze_data_prefix(args.data_path)
        setattr(args, "data_prefixes", prefixes)
        setattr(args, "data_weights", weights)
    elif args.train_weighted_split_paths is not None and len(args.train_weighted_split_paths[0]) > 1:
        paths = args.train_weighted_split_paths[0]
        weights = args.train_weighted_split_weights[0]
        data_prefix = [j for i in [[w,p] for w,p in zip(weights, paths)] for j in i]
        prefixes, weights = analyze_data_prefix(data_prefix)
        setattr(args, "data_prefixes", prefixes)
        setattr(args, "data_weights", weights)
    else:
        setattr(args, "data_prefixes", None)
        setattr(args, "data_weights", None)

    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])
    print_rank_0('training ...')

    iteration = 0
    if args.do_train and args.train_iters > 0:
        iteration = train(forward_step_func,
                          model, optimizer, lr_scheduler,
                          train_data_iterator, valid_data_iterator)
    print_datetime('after training is done')

    if args.do_valid:
        names = args.valid_weighted_split_names
        names = names if names is not None else ['valid'] * len(valid_data_iterator)
        for iterator, name in zip(valid_data_iterator, names):
            prefix = 'the end of training for val data'
            evaluate_and_print_results(prefix, forward_step_func,
                                       iterator, model,
                                       iteration, False, data_group_name=name)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        names = args.test_weighted_split_names
        names = names if names is not None else ['test'] * len(test_data_iterator)
        for iterator, name in zip(test_data_iterator, names):
            evaluate_and_print_results(prefix, forward_step_func,
                                       iterator, model,
                                       0, True, data_group_name=name)

    codecarbon_tracker_stop()


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process
        )


    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # # Print number of parameters.
    # Moved to `train` with extras
    # if mpu.get_data_parallel_rank() == 0:
    #     print('Number of parameters on tensor={}, pipeline={}: {}'.format(
    #         mpu.get_tensor_model_parallel_rank(),
    #         mpu.get_pipeline_model_parallel_rank(),
    #         sum([sum([p.ds_numel if hasattr(p,'ds_id') else p.nelement() for p in model_module.parameters()])
    #              for model_module in model])), flush=True)
    #     torch.distributed.barrier()
    # else:
    #     torch.distributed.barrier()

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = [torchDDP(model_module, device_ids=[i], output_device=i,
                          process_group=mpu.get_data_parallel_group())
                 for model_module in model]
        return model

    if args.DDP_impl == 'local':
        model = [LocalDDP(model_module,
                          args.accumulate_allreduce_grads_in_fp32,
                          args.use_contiguous_buffers_in_ddp)
                 for model_module in model]
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def sync_hp_to_lp(optimizer):

    optimizer.update_lp_params()

    # for n,p in model.named_parameters():
    #     print(n)

    #     if p._hp_mapping is not None:
    #         #print(f'rank {rank} fixing hp for input_layernorm')
    #         #p._hp_mapping.update_hp()

    #         hp = p._hp_mapping.hp_fragment



    #         torch.distributed.all_reduce(hp, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())

    #         # 3. optim states
    #         for key in ['exp_avg', 'exp_avg_sq']:
    #             optim_state_fragment = p._hp_mapping.get_optim_state_fragment(key)
    #             #print(f'rank {rank} before reduce optim state fragment {key} = {optim_state_fragment}')
    #             torch.distributed.all_reduce(optim_state_fragment, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group())
    #             #print(f'rank {rank} after reduce optim state fragment {key} = {optim_state_fragment}')


def setup_layer_groups_for_training(model, args, group_idx):
    transformer_layers = extract_pipetransformer_layers(model)
    num_layers = len(transformer_layers)
    num_groups = args.num_groups
    group_size, remainder = divmod(num_layers, num_groups)
    groups = []
    start = 0
    for i in range(num_groups):
        end = start + group_size + (1 if i < remainder else 0)
        groups.append(transformer_layers[start:end])
        start = end
    update_requires_grad(model, transformer_layers, groups, group_idx)

def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, Float16Module))

    if args.inference:
        optimizer = None
        lr_scheduler = None
    else:
        optimizer = get_megatron_optimizer(unwrapped_model)
        lr_scheduler = get_learning_rate_scheduler(optimizer)


    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        #pp = mpu.get_pipeline_model_parallel_world_size()

        import json
        import io
        with io.open(args.deepspeed_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        if args.universal_checkpoint:
            config["checkpoint"] = {"load_universal": True}

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config
        )
        
        assert model.fp16_enabled() == args.fp16, "megatron fp16 config does not match deepspeed"
        assert model.bfloat16_enabled() == args.bf16, "megatron bf16 config does not match deepspeed"

        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]

    if args.load is not None:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        torch.distributed.barrier()
        timers('load-checkpoint').start()
        if args.trainblock:
            load_arg='load'
            load_dir = getattr(args, load_arg)
            groupid_filename = get_checkpoint_groupid_filename(load_dir)
            if  os.path.isfile(groupid_filename):
                with open(groupid_filename, 'r') as f:
                    idstring = f.read().strip()
                    group_idx = int(idstring)
                setup_layer_groups_for_training(model, args, group_idx)
            else:
                group_idx=0
            args.group_idx = group_idx
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
        torch.distributed.barrier()
        timers('load-checkpoint').stop()
        timers.log(['load-checkpoint'])


        # hp -> lp
        if args.deepspeed and args.universal_checkpoint:
            sync_hp_to_lp(optimizer)


    else:
        args.iteration = 0

    # tp_rank = mpu.get_tensor_model_parallel_rank()
    # pp_rank = mpu.get_pipeline_model_parallel_rank()
    # dp_rank = mpu.get_data_parallel_rank()
    # for n,p in model[0].named_parameters():
    #     if 'word_embeddings.weight' not in n:
    #         continue 
    #     if tp_rank == 0 and pp_rank == 0:
    #         print(f"{tp_rank=}{pp_rank=}{dp_rank=} bf16 {n=} {p[:10]=}")
    #         if p._hp_mapping is not None:
    #             hp = p._hp_mapping.hp_fragment
    #             print(f'{tp_rank=}{pp_rank=}{dp_rank=} fp32 {n=} {hp[:10]=}')

    #     if tp_rank == 0 and pp_rank == mpu.get_pipeline_model_parallel_world_size() - 1:
    #         print(f"{tp_rank=}{pp_rank=}{dp_rank=} bf16 {n=} {p[:10]=}")
    #         if p._hp_mapping is not None:
    #             hp = p._hp_mapping.hp_fragment
    #             print(f'{tp_rank=}{pp_rank=}{dp_rank=} fp32 {n=} {hp[:10]=}')


    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, lr_scheduler


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()
    if args.deepspeed:
        assert isinstance(model[0], deepspeed.PipelineEngine), model
        loss = model[0].train_batch(data_iter=data_iterator)
        skipped_iter = 0
        grad_norm = model[0].get_global_grad_norm()
        num_zeros_in_grad = 0
        return {'lm loss' : loss}, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    if not args.deepspeed:
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_ddp:
            for partition in model:
                partition.zero_grad_buffer()
        else:
            optimizer.zero_grad()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
                'number of microbatches is not divisible by pipeline-parallel ' \
                'size when using interleaved schedule'
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model,
        optimizer, timers, forward_only=False)

    # All-reduce if needed.
    if not args.deepspeed and args.DDP_impl == 'local':
        timers('backward-params-all-reduce').start()
        for model_module in model:
            model_module.allreduce_gradients()
        timers('backward-params-all-reduce').stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    timers('backward-embedding-all-reduce').start()
    if not args.deepspeed:
        if (mpu.is_pipeline_first_stage(ignore_virtual=True) or
            mpu.is_pipeline_last_stage(ignore_virtual=True)) and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = model[0]
            elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = model[-1]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module))

            if unwrapped_model.share_word_embeddings:
                word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                if args.DDP_impl == 'local':
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
    timers('backward-embedding-all-reduce').stop()

    # Update parameters.
    timers('optimizer').start()
    if args.deepspeed:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        model[0].step(lr_kwargs={'increment': increment})
        update_successful = model[0].was_step_applied()
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        grad_norm = None
        num_zeros_in_grad = None
    else:
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            lr_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward-compute')
    add_to_logging('forward-recv')
    add_to_logging('forward-send')
    add_to_logging('forward-backward-send-forward-backward-recv')
    add_to_logging('backward-compute')
    add_to_logging('backward-recv')
    add_to_logging('backward-send')
    add_to_logging('backward-send-forward-recv')
    add_to_logging('backward-send-backward-recv')
    add_to_logging('backward-params-all-reduce')
    add_to_logging('backward-embedding-all-reduce')
    add_to_logging('optimizer-copy-to-main-grad')
    add_to_logging('optimizer-unscale-and-check-inf')
    add_to_logging('optimizer-clip-main-grad')
    add_to_logging('optimizer-copy-main-to-model-params')
    add_to_logging('optimizer')
    add_to_logging('batch-generator')

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    if writer and (iteration % args.tensorboard_log_interval == 0) and \
       is_last_rank():
        writer.add_scalar('steps-vs-samples/y=steps,x=samples', iteration, args.consumed_train_samples)
        writer.add_scalar('steps-vs-samples/y=samples,x=steps', args.consumed_train_samples, iteration)
        writer.add_scalar('steps-vs-tokens/y=steps,x=tokens', iteration, args.consumed_train_tokens)
        writer.add_scalar('steps-vs-tokens/y=tokens,x=steps', args.consumed_train_tokens, iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate/learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate/learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            writer.add_scalar('learning-rate/learning-rate vs tokens', learning_rate,
                              args.consumed_train_tokens)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar('batch-size/batch-size', batch_size, iteration)
            writer.add_scalar('batch-size/batch-size vs samples', batch_size,
                              args.consumed_train_samples)
        for key in loss_dict:
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            writer.add_scalar(f"lm-loss-training/{key}" + ' vs tokens', loss_dict[key],
                              args.consumed_train_tokens)

            writer.add_scalar(f"lm-loss-training/{key}" + ' vs gigaflos (without embeddings)', loss_dict[key],
                              args.gigaflos_no_embeds)
        if args.log_loss_scale_to_tensorboard and args.fp16:
            writer.add_scalar('loss-scale/loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale/loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            writer.add_scalar('loss-scale/loss-scale vs tokens', loss_scale,
                              args.consumed_train_tokens)
        if grad_norm is not None:
            writer.add_scalar('grad-norm/grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm/grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            writer.add_scalar('grad-norm/grad-norm vs tokens', grad_norm,
                              args.consumed_train_tokens)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros/num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros/num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            writer.add_scalar('num-zeros/num-zeros vs tokens', num_zeros_in_grad,
                              args.consumed_train_tokens)
        if params_norm is not None:
            writer.add_scalar('params-norm/params-norm', params_norm, iteration)
            writer.add_scalar('params-norm/params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            writer.add_scalar('params-norm/params-norm vs tokens', params_norm,
                              args.consumed_train_tokens)
        if args.curriculum_learning:
            writer.add_scalar('curriculum_seqlen', args.curriculum_seqlen,
                              iteration)

        # It's very questionable what this data contributes, other than huge unstripped file paths
        # as keys and hundreds of TB boards that make the TB files very bloated. So disabling for now.
        #
        # if args.data_weights is not None:
        #     for prefix, weight in zip(args.data_prefixes, args.data_weights):
        #         name = prefix.split(",")[-1]
        #         writer.add_scalar(f'samples-per-dataset/{name}', args.consumed_train_samples * weight, args.consumed_train_samples)
        #         writer.add_scalar(f'steps-per-dataset/{name}', iteration * weight, iteration)
        #         writer.add_scalar(f'tokens-per-dataset/{name}', args.consumed_train_tokens * weight, args.consumed_train_tokens)

        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration,
                         normalizer=total_iterations)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed()
        elapsed_time_per_iteration = elapsed_time / total_iterations

        seq_len = args.curriculum_seqlen if args.curriculum_learning else args.seq_length
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        vocab_size = args.padded_vocab_size

        # Compute throughput.
        samples_per_sec = batch_size / elapsed_time_per_iteration
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size

        # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
        # https://arxiv.org/pdf/2104.04473.pdf).
        # The factor of 4 is when used with activation check-pointing,
        # otherwise it will be 3, but for 200B model, activation check-pointing will always be on.
        checkpoint_activations_factor = 4 if args.checkpoint_activations else 3
        # GLU activations double the hidden states in the upscaling feed-forward in each transformer layer
        # This leads to 16bsh^2 instead of 8bsh^2 per first feed-forward layer in MLP, thus we increase the coefficient by 8.
        # Refer to https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/283#issue-1260805063 for more details.
        coefficient = 32 if args.glu_activation else 24
        flops_per_iteration = (coefficient * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
        tflops = flops_per_iteration / (elapsed_time_per_iteration * args.world_size * (10**12))

        # only the last rank process has a non-None _GLOBAL_TENSORBOARD_WRITER
        if writer and is_last_rank():
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time/iteration-time',
                                  elapsed_time_per_iteration, iteration)
                writer.add_scalar('iteration-time/iteration-time vs samples',
                                  elapsed_time_per_iteration, args.consumed_train_samples)
                writer.add_scalar('iteration-time/iteration-time vs tokens',
                                  elapsed_time_per_iteration, args.consumed_train_tokens)
                writer.add_scalar('iteration-time/samples per second',
                                  samples_per_sec, args.iteration)
                writer.add_scalar('iteration-time/samples per second per replica',
                                  samples_per_sec_per_replica, args.iteration)
                writer.add_scalar('iteration-time/tokens per second',
                                  tokens_per_sec, args.iteration)
                writer.add_scalar('iteration-time/tokens per second per replica',
                                  tokens_per_sec_per_replica, args.iteration)
                writer.add_scalar('iteration-time/TFLOPs per gpu (estimated)',
                                  tflops, args.iteration)

        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' consumed tokens: {:12d} |'.format(
            args.consumed_train_tokens)
        log_string += ' elapsed time per iteration (s): {:.2f} |'.format(
            elapsed_time_per_iteration)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        if args.fp16:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        if args.curriculum_learning:
            log_string += ' curriculum seqlen: {:5d} |'.format(args.curriculum_seqlen)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
        log_string += ' TFLOPs: {:.2f} |'.format(tflops)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)
        flops_calculator(model, args, elapsed_time)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers('save-checkpoint').start()
    save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers('save-checkpoint').stop()
    timers.log(['save-checkpoint'])

def extract_pipetransformer_layers(model, transformer_layers=None):
    if transformer_layers is None:
        transformer_layers = []

    for name, submodule in model[0].named_children():
        if name == "module":
            for child_name, child_module in submodule.named_children():
                if child_module.__class__.__name__ == "ParallelTransformerLayerPipe":
                    transformer_layers.append(child_module)
    return transformer_layers

def update_requires_grad(model, transformer_layers, groups, group_idx):
    args = get_args()

    for layer in transformer_layers:
        for param in layer.parameters():
            param.requires_grad = False

    current_group = groups[group_idx]
    for layer in current_group:
        for param in layer.parameters():
            param.requires_grad = True

    params_to_train = [p for p in model[0].parameters() if p.requires_grad]

    optimizer = model[0].optimizer

    if not args.fp16 and not args.bf16:
        # Clear old parameter groups and release states
        optimizer.param_groups.clear()
        optimizer.state.clear()

        # Add new parameter group
        optimizer.add_param_group({"params": params_to_train})
    elif args.fp16:
        new_param_group = {"params": list(params_to_train)}
        optimizer.optimizer.param_groups.clear()
        optimizer.optimizer.state.clear()
        optimizer.optimizer.add_param_group(new_param_group)

        optimizer.initialize_optimizer_states()
        # optimizer.state.clear()
        # real_optimizer = optimizer.optimizer 
        # real_optimizer.state.clear()
        # optimizer.initialize_optimizer_states()
        # optimizer.has_executed_step = False
    elif args.bf16:
        new_param_group = {"params": list(params_to_train)}

        optimizer.bf16_groups.clear()
        optimizer.bf16_groups_flat.clear()
        optimizer.bf16_partitioned_groups.clear()

        optimizer.fp32_groups_flat_partition.clear()
        optimizer.fp32_groups_gradients.clear()
        optimizer.fp32_groups_gradient_dict.clear()
        optimizer.fp32_groups_gradients_flat.clear()
        optimizer.fp32_groups_actual_gradients_flat.clear()
        optimizer.fp32_groups_gradient_flat_partition.clear()
        optimizer.fp32_groups_has_gradients.clear()
        optimizer.group_paddings.clear()

        optimizer.optimizer.param_groups.clear()
        optimizer.optimizer.state.clear()
        optimizer.optimizer.add_param_group(new_param_group)
        
        optimizer._setup_for_real_optimizer()

    #     if base_optimizer.param_groups:
    #         base_group = base_optimizer.param_groups[0]
    #         new_group = {k: v for k, v in base_group.items() if k != 'params'}
    #         new_group['params'] = params_to_train
    #     else:
    #         new_group = {
    #             'params': params_to_train,
    #             'bias_correction': True,
    #             'betas': (0.9, 0.95),
    #             'eps': 1e-8,
    #             'weight_decay': 1e-5,
    #             'step': 1,
    #         }

    #     optimizer._set_param_groups([new_group])

    #     keep_ids = {id(p) for p in params_to_train}
    #     keys_to_delete = [k for k in base_optimizer.state if id(k) not in keep_ids]
    #     for k in keys_to_delete:
    #         del base_optimizer.state[k]

    #     optimizer.fp16_groups = [list(params_to_train)]

    #     optimizer.fp32_groups_flat = [_flatten_dense_tensors([
    #         p.data.float() for p in params_to_train
    #     ])]
    #     optimizer.fp16_groups_flat = [_flatten_dense_tensors([
    #         p.data for p in params_to_train
    #     ])]

def train(forward_step_func, model, optimizer, lr_scheduler,
          train_data_iterator, valid_data_iterator):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    if args.rank == 0:
        print("Number of parameters: [tensor rank - pipeline rank] w/ and w/o embeddings:")
    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        preamble = f"[{tp_rank:0>3d}-{pp_rank:0>3d}]"
        print(f"{preamble} {get_parameters_in_billions(model):.4f}B / {get_parameters_in_billions(model, exclude_embeddings=True):.4f}B", flush=True)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

    # Write args to tensorboard
    write_args_to_tensorboard()
    log_restart_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()
    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    
    if args.trainblock:
        print_rank_0("BCD start....")
        transformer_layers = extract_pipetransformer_layers(model)
        num_layers = len(transformer_layers)
        num_groups = args.num_groups
        assert num_groups <= len(transformer_layers), \
            f"num_groups ({num_groups}) cannot exceed number of transformer layers ({len(transformer_layers)})"
        group_size, remainder = divmod(num_layers, num_groups)

        groups = []
        start = 0
        for i in range(num_groups):
            if i < remainder:
                end = start + group_size + 1
            else:
                end = start + group_size
            groups.append(transformer_layers[start:end])
            start = end
        # print_rank_0(groups)
        update_requires_grad(model,transformer_layers,groups,args.group_idx)

    timers('interval-time').start()
    print_datetime('before the start of training step')
    report_memory_flag = True

    # flush intervals prior to current iteration
    if args.skip_train_iteration_range is not None:
        ends = [end for start, end in args.skip_train_iteration_range]
        index = bisect.bisect_left(ends, iteration)
        for _ in range(index):
            args.skip_train_iteration_range.popleft()

    prev_lm_loss = None 
    for iteration in range(args.train_iters):
    # while iteration < args.train_iters:
        if (
            # train_data_iterator is not None
            args.skip_train_iteration_range is not None
            and len(args.skip_train_iteration_range) > 0
            and args.skip_train_iteration_range[0][0] <= iteration + 1 <= args.skip_train_iteration_range[0][1]
        ):
            start, end = args.skip_train_iteration_range.popleft()
            print_rank_0(f"Skipped iterations {start} to {end} due to --skip-train-iteration-range flag.")
            iteration_for_skipping = args.iteration
            while iteration_for_skipping + 1 <= end:
                try:
                    _ = next(train_data_iterator)
                except TypeError:
                    pass
                iteration_for_skipping += 1
            continue

        if found_kill_switch():
            save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
            print_datetime(f"Detected kill switch at {args.kill_switch_path}. Exiting")
            sys.exit()

        update_num_microbatches(args.consumed_train_samples)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = mpu.get_data_parallel_world_size() * \
                                args.micro_batch_size * \
                                get_num_microbatches()
            model[0].set_train_batch_size(global_batch_size)

        if args.curriculum_learning and \
            args.pipeline_model_parallel_size >= 1:
            args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                    args.iteration + 1)
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       lr_scheduler)
        # iteration += 1
        args.iteration = iteration
        new_samples = mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()
        args.consumed_train_samples += new_samples
        if args.curriculum_learning:
            args.consumed_train_tokens += new_samples * args.curriculum_seqlen
        else:
            args.consumed_train_tokens += new_samples * args.seq_length
        args.gigaflos_no_embeds += (6 * new_samples * args.seq_length * get_parameters_in_billions(model, exclude_embeddings=True))

        # Logging.
        loss_scale = None
        if args.fp16:
            if args.deepspeed:
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad,
                                          model)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            timers('interval-time').stop()
            prefix = 'iteration {}'.format(iteration)
            names = args.valid_weighted_split_names
            names = names if names is not None else ['valid'] * len(valid_data_iterator)
            for iterator, name in zip(valid_data_iterator, names):
                evaluate_and_print_results(prefix, forward_step_func,
                                           iterator, model,
                                           iteration, False, data_group_name=name)
            timers('interval-time').start()

        # Checkpointing
        saved_checkpoint = False
        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            timers('interval-time').stop()
            save_checkpoint_and_time(iteration, model, optimizer,
                                     lr_scheduler)
            saved_checkpoint = True
            timers('interval-time').start()

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         lr_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            sys.exit()
        if args.trainblock:
            lm_loss = loss_dict['lm loss'].item()
            if prev_lm_loss is not None and abs(prev_lm_loss - lm_loss) < 0.5:
                args.group_idx = (args.group_idx + 1) % num_groups  # 切换到下一个 group
                print_rank_0(f"Loss difference < 0.00005 at iteration {iteration},change group {args.group_idx}...")
                update_requires_grad(model, transformer_layers, groups, args.group_idx)

            # 更新上一轮的 loss
            prev_lm_loss = lm_loss
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            max_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

            print_rank_0(f"显存使用情况(MB) | "
                        f"allocated: {allocated:.2f} | "
                        f"max allocated: {max_allocated:.2f} | "
                        f"reserved: {reserved:.2f} | "
                        f"max reserved: {max_reserved:.2f}")
    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.curriculum_learning and \
        args.pipeline_model_parallel_size >= 1:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            model[0].reset_activation_shape()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                if args.virtual_pipeline_model_parallel_size is not None:
                    forward_backward_func = forward_backward_pipelining_with_interleaving
                else:
                    forward_backward_func = forward_backward_pipelining_without_interleaving
            else:
                forward_backward_func = forward_backward_no_pipelining

            if args.deepspeed:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{'lm loss' : loss}] * get_num_microbatches()
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func, data_iterator, model, optimizer=None,
                    timers=None, forward_only=True)

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += mpu.get_data_parallel_world_size() \
                                           * args.micro_batch_size \
                                           * get_num_microbatches()
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    if args.curriculum_learning and \
        args.pipeline_model_parallel_size >= 1:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
            args.iteration + 1)
        if args.curriculum_seqlen < args.seq_length:
            model[0].reset_activation_shape()

    return total_loss_dict

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False, **kwargs):
    """Helper function to evaluate and dump results on screen."""


    args = get_args()
    writer = get_tensorboard_writer()

    ds_name = kwargs.get("data_group_name", None)
    # print corresponding dataset name (used for multiple validation datasets)
    tf_plot_prefix = f"lm-loss-validation/{ds_name}" if ds_name else "lm-loss-validation"

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = '{} loss at {} | '.format(ds_name, prefix) if ds_name is not None\
        else 'validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and is_last_rank():
            writer.add_scalar(f'{tf_plot_prefix}/{key} validation',
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar(f'{tf_plot_prefix}/{key} validation vs samples',
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            writer.add_scalar(f'{tf_plot_prefix}/{key} validation vs tokens',
                              total_loss_dict[key].item(),
                              args.consumed_train_tokens)
            writer.add_scalar(f'{tf_plot_prefix}/{key} validation vs gigaflos (without embeddings)',
                              total_loss_dict[key].item(),
                              args.gigaflos_no_embeds)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(f'{tf_plot_prefix}/{key} validation ppl', ppl,
                                  iteration)
                writer.add_scalar(f'{tf_plot_prefix}/{key} validation ppl vs samples',
                                  ppl, args.consumed_train_samples)
                writer.add_scalar(f'{tf_plot_prefix}/{key} validation ppl vs tokens',
                                  ppl, args.consumed_train_tokens)
                writer.add_scalar(f'{tf_plot_prefix}/{key} validation ppl vs gigaflos (without embeddings)',
                                  ppl, args.gigaflos_no_embeds)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """XXX"""
    args = get_args()

    (train_dataloader, valid_dataloaders, test_dataloaders) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    # it's possible that train was run, but not eval and it's valid if
    # args.consumed_valid_samples == 0
    # TODO: eval_interval could have changed between runs, so this might still be wrong
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:
        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(train_val_test_num_samples)

        # if dataloading option is not 2 convert to list to allow
        # same interface for multiple data groups
        # for validation and testing in option 2
        if type(train_ds) != list and train_ds is not None:
            train_ds = [train_ds]
        if type(valid_ds) != list and valid_ds is not None:
            valid_ds = [valid_ds]
        if type(test_ds) != list and test_ds is not None:
            test_ds = [test_ds]

        # Build dataloders.
        assert len(train_ds) == 1, "only one training dataset group is allowed"

        # train_dataloader is a single item while valid_dataloaders
        # and test_dataloaders are arrays
        train_dataloader = build_pretraining_data_loader(
            train_ds[0], args.consumed_train_samples)

        # We collapse None and empty list as both should mean we don't run validation
        # args.consumed_valid_samples accumulates the sum of valid steps for every dataset, which are all equal
        #
        # XXX: we get a deadlock in the dataloader on multi-dataset eval, after the first dataset,
        # possibly due to this bug in pytorch https://github.com/pytorch/pytorch/pull/25158. Using
        # num_workers=0 to work around it - the training can't use that since it impacts throughput
        # by a few percent
        valid_dataloaders = [build_pretraining_data_loader(d, args.consumed_valid_samples // len(valid_ds), num_workers=args.valid_num_workers)
                            for d in valid_ds] \
                            if valid_ds is not None else []
        # We collapse None and empty list as both should mean we don't run test
        test_dataloaders = [build_pretraining_data_loader(d, 0) for d in test_ds] \
                            if test_ds is not None else []

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0 and not args.eval_only

        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([
            int(do_train),
            len(valid_dataloaders) if args.eval_iters > 0 else 0, # eval_iters == 0 is equivalent to having no validation
            len(test_dataloaders) if args.eval_iters > 0 else 0, # eval_iters == 0 is equivalent to having no test
        ])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    num_valid_ds = flags[1].item()
    num_test_ds = flags[2].item()
    assert num_test_ds >= 0
    assert num_valid_ds >= 0
    args.do_valid = num_valid_ds > 0
    args.do_test = num_test_ds > 0

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type in ['single'] \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloaders is not None:
        valid_data_iterators = [iter(vdl) if dl_type in ['single'] \
                              else iter(cyclic_iter(valid_dataloaders))
                                 for vdl in valid_dataloaders]
    else:
        valid_data_iterators = [None] * num_valid_ds

    if test_dataloaders is not None:
        test_data_iterators = [iter(tdl) if dl_type in ['single'] \
                             else iter(cyclic_iter(test_dataloaders))
                            for tdl in test_dataloaders]
    else:
        test_data_iterators = [None] * num_test_ds

    return train_data_iterator, valid_data_iterators, test_data_iterators
