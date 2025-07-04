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

"""Megatron arguments."""

import argparse
import collections
import os
import re
import time

import torch
import deepspeed

from megatron.enums import PositionEmbeddingType
import megatron
from megatron.logging import log_levels


def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_biencoder_args(parser)
    parser = _add_vit_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_zero_args(parser)
    parser = _add_memoryopt_args(parser)
    parser = _add_activation_checkpoint_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    parser = deepspeed.add_config_arguments(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)
    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    assert args.world_size % model_parallel_size == 0, 'world size is not'\
        ' divisible by tensor parallel size ({}) times pipeline parallel ' \
        'size ({})'.format(args.world_size, args.tensor_model_parallel_size,
                           args.pipeline_model_parallel_size)
    args.data_parallel_size = args.world_size // model_parallel_size
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)

    # --data-path and --train-weighted-splits-paths
    message = "Data loading Mode 1: --data-path and --split "\
            "and Mode 2: --(train|valid|test)-weighted-split-paths"\
            "are mutually exclusive i.e. cannot be set together."

    if args.data_path:
        assert args.train_weighted_split_paths is None, message
        setattr(args, "valid_weighted_split_names", None)
        setattr(args, "valid_weighted_split_weights", None)
        setattr(args, "valid_weighted_split_splits", None)

        setattr(args, "test_weighted_split_names", None)
        setattr(args, "test_weighted_split_weights", None)
        setattr(args, "test_weighted_split_splits", None)

        # args.split default value in the args is None it is set here in order
        # to check that it does not to overlap with the 2nd mode of data loading
        if args.split is None:
            args.split = "969, 30, 1"

    if args.train_weighted_split_paths or args.valid_weighted_split_paths or \
                args.test_weighted_split_paths:
        assert args.data_path is None and args.split is None, message



    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers is not divisible by number of layers per virtual ' \
            'pipeline stage'
        args.virtual_pipeline_model_parallel_size = \
            (args.num_layers // args.pipeline_model_parallel_size) // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # If we do accumulation and all-reduces in fp32, we need to have
    # local DDP and we should set the use-contiguous-buffers-in-ddp.
    if args.accumulate_allreduce_grads_in_fp32:
        assert args.DDP_impl == 'local'
        args.use_contiguous_buffers_in_ddp = True

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.consumed_train_tokens = 0
    args.gigaflos_no_embeds = 0

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.position_embedding_type == PositionEmbeddingType.absolute or args.position_embedding_type == PositionEmbeddingType.alibi:
        assert args.max_position_embeddings is not None
        if args.seq_length is not None:
            assert args.max_position_embeddings >= args.seq_length
        if args.decoder_seq_length is not None:
            assert args.max_position_embeddings >= args.decoder_seq_length
    else:
        assert args.max_position_embeddings is None

    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'
    # Activation checkpointing.
    if args.distribute_checkpointed_activations:
        assert args.checkpoint_activations, \
            'for distribute-checkpointed-activations to work you '\
            'need to enable checkpoint-activations'

    args.curriculum_learning = False

    # Activation function
    if args.glu_activation is not None and args.bias_gelu_fusion:
        raise ValueError("if glu-activation is used, please set --no-bias-gelu-fusion")

    # Skip train iterations
    if args.skip_train_iteration_range is not None:
        args.skip_train_iteration_range = [
            list(map(int, range_.split("-"))) for range_ in args.skip_train_iteration_range
        ]
        args.skip_train_iteration_range.sort()
        skip_train_iteration_range = collections.deque()
        for range_ in args.skip_train_iteration_range:
            if len(range_) == 2:
                start, end = range_
                assert end >= start, \
                "end of skip range cannot be smaller than start of skip range"
                # merge overlapping intervals (e.g. 1-5 2-6 -> 1-6)
                if not skip_train_iteration_range:
                    skip_train_iteration_range.append([start, end])
                elif skip_train_iteration_range[-1][1] >= start:
                    skip_train_iteration_range[-1][1] = max(end, skip_train_iteration_range[-1][1])
                else:
                    skip_train_iteration_range.append([start, end])
            else:
                raise ValueError(
                    "skip train iterations should be specified as two numbers, i.e. start-end"
                )
        args.skip_train_iteration_range = skip_train_iteration_range

    if args.use_bnb_optimizer:
        try:
            import bitsandbytes as bnb
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install bitsandbytes from https://github.com/facebookresearch/bitsandbytes.")

    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))

        if args.log_path is not None:
            with open(os.path.join(args.log_path,f'args_{time.strftime("%Y-%m-%dT%H:%M:%S")}.txt'), 'w') as f:
                for arg in sorted(str_list, key=lambda x: x.lower()):
                    f.write(arg+"\n")
                    print(arg, flush=True)
        else:
            for arg in sorted(str_list, key=lambda x: x.lower()):
                print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to this value.'
                       'This value must be greater than the initial size of the tokenizer'
                       ', needs to be divisible by TP size and `make-vocab-size-divisible-by`.')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--sync-tp-duplicated-parameters', action='store_true',
                       help='Force syncing duplicated params across TP ranks in forward. '
                       'This is a workaround for an unresolved bug leading to TP ranks '
                       'getting out of sync with each other.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--embed-layernorm', action='store_true',
                       help='use layernorm for embedding')
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--position-embedding-type', type=lambda x: PositionEmbeddingType[x],
                       choices=list(PositionEmbeddingType),
                       default=PositionEmbeddingType.absolute,
                       help='Define position embedding type ("absolute" | "rotary" | "alibi"). "absolute" by default.'
                       )
    group.add_argument('--glu-activation', type=str,
                       choices=megatron.model.glu_activations.GLU_ACTIVATIONS.keys(),
                       help='GLU activations to use.'
                       )

    group.add_argument('--kill-switch-path', type=str,
                       help='path to look for a kill switch, which if found will automatically exit the program'
                       )


    group.add_argument('--log-level', type=str, choices=list(log_levels.keys()),
                       help="Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug', "
                       "'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the "
                       "application set the level."
                       )
    group.add_argument('--log-level-replica', type=str, choices=list(log_levels.keys()),
                       help="Logger log level to use on replicas. Same choices as ``log_level``"
                       )
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-params-norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log-num-zeros-in-grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--tensorboard-log-interval', type=int, default=1,
                       help='Report to tensorboard interval.')
    group.add_argument('--tensorboard-queue-size', type=int, default=1000,
                       help='Size of the tensorboard queue for pending events '
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--log-batch-size-to-tensorboard', action='store_true',
                       help='If set, write batch-size to tensorboard.')
    group.add_argument('--no-log-learnig-rate-to-tensorboard',
                       action='store_false',
                       help='Disable learning rate logging to tensorboard.',
                       dest='log_learning_rate_to_tensorboard')
    group.add_argument('--no-log-loss-scale-to-tensorboard',
                       action='store_false',
                       help='Disable loss-scale logging to tensorboard.',
                       dest='log_loss_scale_to_tensorboard')
    group.add_argument('--log-validation-ppl-to-tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size increment> '
                       '                      <ramp-up samples> '
                       'For example: '
                       '   --rampup-batch-size 16 8 300000 '
                       '   --global-batch-size 1024 '
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase '
                       'the batch size linearly to 1024. In each interval '
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--distribute-checkpointed-activations',
                       action='store_true',
                       help='If set, distribute checkpointed activations '
                       'across model parallel group.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    
    group.add_argument('--trainblock', action='store_true',
                    help='open block train')
    group.add_argument('--num-groups', type=int, default=None,
                       help="submit transformer")
    group.add_argument('--switch-step', type=int, default=None,
                       help="trans freezn group ")
    
    group.add_argument('--train-tokens', type=int, default=None,
                       help='Total number of tokens to train over all '
                       'training runs.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer function')
    group.add_argument('--use-bnb-optimizer', action='store_true',
                       help='Use bitsandbytes optimizer for efficient training,'
                       'please refer https://github.com/facebookresearch/bitsandbytes.',
                       dest='use_bnb_optimizer')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--cpu-optimizer', action='store_true',
                       help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                       help='Use Torch Adam as optimizer on CPU.')
    group.add_argument('--codecarbon-dir', type=str, default=None,
                       help='Write CodeCarbon logs to this directory.')
    group.add_argument('--eval-only', type=bool, required=False,
                       help='If set to True, no train step will be performed.'
                       'and only the evaluation on the `valid` and `test` sets '
                       'will be performed' )
    group.add_argument('--skip-train-iteration-range', type=str, nargs='+', default=None,
                       help='Iteration ranges to skip. The values are one or more dash-separated ranges. e.g., 101-200 251-300.')
    group.add_argument('--inference', action='store_true',
                       help='Very basic inference mode: not allocating optim/lr - requires ZERO_STAGE=0')
    group.add_argument('--abort-on-unmet-fused-kernel-constraints', action='store_true',
                       help="If set to True, the program will abort if the constraints for loading a fused kernel aren't met")
    group.add_argument('--pp-partition-method', type=str, default=None,
                       help="Use to override the pipeline stages partitioning method. e.g., 'type:transformer|embedding'")

    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-decay-tokens', type=int, default=None,
                       help='number of tokens to decay learning rate over,'
                       ' If not None will override iter/sample-based decay')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-lr-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-lr-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    group.add_argument('--universal-checkpoint', action='store_true',
                        help='Loading a universal format checkpoint.')

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--save-total-limit', type=int, default=2,
                       help='save total limit.')
    group.add_argument('--metric-for-best-model', type=str, default=None,
                       help='metric for best model')
    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')

    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scale.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--no-query-key-layer-scaling', action='store_false',
                       help='Do not scale Q * K^T by 1 / layer-number.',
                       dest='apply_query_key_layer_scaling')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32. '
                       'This flag is ignored unless '
                       '--no-query-key-layer-scaling is specified.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--use-contiguous-buffers-in-ddp', action='store_true',
                       help='If set, use contiguous buffer in DDP. Note that '
                       'this option only works woth local DDP.' )
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU' )
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')


    # option 1 for data loading  (mutually exclusive with option2)
    group.add_argument('--data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')

    group.add_argument('--split', type=str, default=None,
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')

    # option 2 for data loading (mutually exclusive with option1)

    # helper class to parse the --xxx-weighted-split-paths
    # note here two args are set: extra valid dataset paths and names
    class parse_data_paths(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):

            if option_string == "--train-weighted-split-paths":
                assert len(values) == 1, 'Only 1 dataset group is allowed to'
                'be passed for the argument --train-weighted-split-paths'

            # make sure string given in the correct format
            err_message = 'Each data group should be input on the following format'
            '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
            'where START < END'
            for v in values:
                # each prefix consists several datasets separated by commas
                prefix = ":".join(v.split(":")[1:]) # remove GIVEN_NAME
                datasets = prefix.split(",")
                # check if each dataset is formatted like `WEIGHT START:END PATH`
                for d in datasets:
                    assert len(d.split()) == 3, err_message
                    start, end = d.split()[1].split(":")
                    assert float(start) < float(end), err_message

            names = [v.split(":")[0] for v in values]

            prefixes = [":".join(v.split(":")[1:]).strip() for v in values]
            weights = [[d.split()[0] for d in p.split(",")] for p in prefixes]
            splits = [[d.split()[1] for d in p.split(",")] for p in prefixes]
            paths = [[d.split()[2] for d in p.split(",")] for p in prefixes]

            # # to keep consistency with Option 1 of data loading (through --data-path)
            # #  paths will contain strings on the following form
            # # "WEIGHTS1 PATH1 WEIGHTS2 PATH2 WEIGHTS3 PATH3" for each dataset group
            # # while data will be parsed in additional arguments below
            # paths_option1_style = []
            # for p, w in zip(paths, weights):
            #   paths_option1_style.append(" ".join([f"{w_i} {p_i}" for p_i, w_i in zip(p,w)]))
            # setattr(args, self.dest, paths_option1_style)
            setattr(args, self.dest, paths)
            setattr(args, self.dest.replace("paths", "weights"), weights)
            setattr(args, self.dest.replace("paths", "splits"), splits)
            setattr(args, self.dest.replace("paths","names"), names)


    group.add_argument('--train-weighted-split-paths', nargs='*', default=None,
                    help='Weights, splits and paths to groups of datasets'
                    'Accepted format: ONE dataset groups could be'
                    'submitted in the following form between double quotes'
                    '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
                    'e.g.: "NAME_ABC: 0.6 0:0.6 A, 0.3 0:1 B, 0.1 0:1 C" '
                    'WEIGHT is used to up and down sample each dataset A,B,C in the group'
                    'START:END indicates the split portion of the dataset',
                    action=parse_data_paths)

    group.add_argument('--valid-weighted-split-paths', nargs='*', default=None,
                    help='Weights, splits and paths to groups of datasets'
                    'Accepted format: one or many dataset groups could be'
                    'submitted in the following form each between double quotes'
                    '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
                    'e.g.: "NAME_ABC: 0.6 0.6:0.8 A, 0.3 0:1 B, 0.1 0:1 C" '
                    '"NAME_CDE: 0.6 0.6:0.8 C, 0.3 0:1 D, 0.1 0:1 E" '
                    'validation will be run on each of those groups independently',
                    action=parse_data_paths)

    group.add_argument('--test-weighted-split-paths', nargs='*', default=None,
                    help='Weights, splits and paths to groups of datasets'
                    'Accepted format: one or many dataset groups could be'
                    'submitted in the following form each between double quotes'
                    '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
                    'e.g.: "NAME_ABC: 0.6 0.6:0.8 A, 0.3 0:1 B, 0.1 0:1 C" '
                    '"NAME_CDE: 0.6 0.6:0.8 C, 0.3 0:1 D, 0.1 0:1 E" '
                    'test will be run on each of those groups independently',
                    action=parse_data_paths)

    class parse_data_paths_path(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            expected_option_strings = ["--train-weighted-split-paths-path", "--valid-weighted-split-paths-path", "--test-weighted-split-paths-path"]
            assert option_string in expected_option_strings, f"Expected {option_string} to be in {expected_option_strings}"

            with open(values, "r") as fi:
                lines = fi.readlines()
                assert len(lines) == 1, f"Got multiple lines {len(lines)} instead of 1 expected"
                assert lines[0][-2:] == "\"\n" and lines[0][0] == "\"", f"Invalid input format, got {lines}"
                values = lines[0][1:-2].split("\" \"")
                weighted_split_paths_dest = re.sub(r"_path$", "", self.dest)
                weighted_split_paths_option = re.sub(r"-path$", "", self.option_strings[0])
                setattr(args, weighted_split_paths_dest, values)
                parse_data_paths(option_strings=[weighted_split_paths_option], dest=weighted_split_paths_dest)(parser, args, values, option_string=weighted_split_paths_option)


    group.add_argument('--train-weighted-split-paths-path', type=str, action=parse_data_paths_path ,default=None)
    group.add_argument('--valid-weighted-split-paths-path', type=str, action=parse_data_paths_path, default=None)
    group.add_argument('--test-weighted-split-paths-path', type=str, action=parse_data_paths_path, default=None)

    group.add_argument('--log-path', type=str, default=None,
                       help='Path to the save arguments file.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help='Maximum encoder sequence length to process.'
                       'This should be exclusive of --seq-length')
    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--retriever-seq-length', type=int, default=256,
                       help='Maximum sequence length for the biencoder model '
                        ' for retriever')
    group.add_argument('--sample-rate', type=float, default=1.0,
                       help='sample rate for training data. Supposed to be 0 '
                            ' < sample_rate < 1')
    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument('--valid-num-workers', type=int, default=2,
                       help="Dataloader number of workers for validation.")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token. Attention between tokens from different documents is null.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')
    group.add_argument('--loss-on-targets-only', action='store_true',
                       help='Mask loss on input sequence.')
    group.add_argument('--reweight-loss-based-on-position-frequency', action="store_true",
                       help='Some objectives require us to sample loss_mask. This might introduce bias towards '
                       'specific positions. This option tries to un-bias the loss by reweighting loss on specific '
                       'positions based on how frequently we train on that position.'
                       'This is mostly used for prefix_lm training')
    group.add_argument("--noise-density", type=float, default=None, help="Span corruption noise density")
    group.add_argument("--mean-noise-span-length", type=int, default=None, help="Span corruption mean noise span length")


    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser


def _add_biencoder_args(parser):
    group = parser.add_argument_group(title='biencoder')

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and '
                        'REALM (paper default: 128)')
    group.add_argument('--biencoder-projection-dim', type=int, default=0,
                       help='Size of projection head used in biencoder (paper'
                        ' default: 128)')
    group.add_argument('--biencoder-shared-query-context-model', action='store_true',
                        help='Whether to share the parameters of the query '
                        'and context models or not')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')
    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence frm DPR paper')

    # training
    group.add_argument('--retriever-report-topk-accuracies', nargs='+', type=int,
                        default=[], help="Which top-k accuracies to report "
                        "(e.g. '1 5 20')")
    group.add_argument('--retriever-score-scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                        'square root of hidden size')

    # faiss index
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')
    group.add_argument('--embedding-path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding'
                        ' data to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')
    return parser


def _add_vit_args(parser):
    group = parser.add_argument_group(title="vit")

    group.add_argument('--num-classes', type=int, default=1000,
                       help='num of classes in vision classificaiton task')
    group.add_argument('--img-dim', type=int, default=224,
                       help='Image size for vision classification task')
    group.add_argument('--num-channels', type=int, default=3,
                       help='Number of channels in input image data')
    group.add_argument('--patch-dim', type=int, default=16,
                       help='patch dimension used in vit')

    return parser


def _add_zero_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('ZeRO configurations', 'configurations')
    group.add_argument("--zero-stage", type=int, default=1.0)
    group.add_argument('--zero-reduce-scatter', action='store_true',
                       help='Use reduce scatter if specified')
    group.add_argument('--zero-contigious-gradients', action='store_true',
                       help='Use contigious memory optimizaiton if specified')
    group.add_argument("--zero-reduce-bucket-size", type=int, default=0.0)
    group.add_argument("--zero-allgather-bucket-size", type=int, default=0.0)
    group.add_argument('--remote-device', type=str, default='none', choices=['none', 'cpu', 'nvme'],
                      help='Remote device for ZeRO-3 initialized parameters.')
    group.add_argument('--use-pin-memory', action='store_true',
                     help='Use pinned CPU memory for ZeRO-3 initialized model parameters.')
    return parser

def _add_memoryopt_args(parser):
    """Memory optimization arguments."""

    group = parser.add_argument_group('Memory optimizations', 'configurations')
    group.add_argument("--scattered-embeddings", action='store_true',
                       help='Save memory by scattering embedding activations. '
                            'Introduces dropout differences across MP configurations.')
    group.add_argument("--split-transformers", action='store_true',
                       help='Save memory by splitting transformer layers into two parts, '
                       'allowing for more frequent activation checkpoint savings.')
    group.add_argument("--memory-centric-tiled-linear", action="store_true",
                       help='Save memory by tiling with deepspeed.zero.TiledLinear.')
    group.add_argument("--tile-factor", type=int, default=1,
                       help='Make all linear layers the same size of [hidden/tile_factor, hidden/tile_factor]. '
                            'Must be enabled with --memory-centric-tiled-linear. '
                            'Example A: if tile_factor=1, the qkv layer [hidden, 3* hidden] would be converted into [1,3] tiles of size [hidden,hidden]. '
                            'Example B: if tile_factor=2, the intermediate layer [4*hidden, hidden] will be converted into [8, 2] tiles of size [hidden/2, hidden/2]. '
                            'Default is 1.')

    return parser

def _add_activation_checkpoint_args(parser):
    group = parser.add_argument_group('Activation Checkpointing',
                                      'Checkpointing Configurations')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--partition-activations', action='store_true',
                       help='partition Activations across GPUs before checkpointing.')
    group.add_argument('--contigious-checkpointing', action='store_true',
                       help='Contigious memory checkpointing for activatoins.')
    group.add_argument('--checkpoint-in-cpu', action='store_true',
                       help='Move the activation checkpoints to CPU.')
    group.add_argument('--synchronize-each-layer', action='store_true',
                       help='does a synchronize at the beginning and end of each checkpointed layer.')
    group.add_argument('--profile-backward', action='store_true',
                       help='Enables backward pass profiling for checkpointed layers.')
    return parser
