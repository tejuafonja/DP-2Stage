import time
import os
import math
import argparse
import random
import logging
import copy
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.accountants.utils import get_noise_multiplier
from transformers import (
    DataCollatorWithPadding,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    CONFIG_MAPPING,
    set_seed,
    BitsAndBytesConfig,
    get_scheduler,
    SchedulerType,
)
from safetensors import safe_open
import safetensors
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    LoKrConfig,
    LNTuningConfig,
    AdaLoraConfig,
    VeraConfig,
)

from utils.misc import mkdir, str2bool
from utils.models import AugmentedBlock
from utils.dataset import LLMtgDataset
from utils.exp_logger import EXPLogger, savefig
from utils.utils import dumpy_config_to_json

import pathlib

import os

PROJECT_DIR = pathlib.Path(__file__).parent.resolve()
os.environ["TRANSFORMERS_CACHE"] = f"{PROJECT_DIR}/./cache/"


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(
        "Finetune transformer models on causal language modeling task."
    )
    
    parser.add_argument("--train_mode", type=str2bool, default=True, help="Enable training mode")
    parser.add_argument("--evaluation_mode", type=str2bool, default=False, help="Enable evaluation mode")
    parser.add_argument("--generation_mode", type=str2bool, default=False, help="Enable generation mode")
    
    parser.add_argument("--weighted_loss", type=float, default=None, help="Weight CEL")

    parser.add_argument("--output_dir", type=str, default="./runs", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1000, help="For reproducibility across different model run with same data split.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    
    parser.add_argument("--serializer", type=str, default="great", help="Table to text conversion method.")
    
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--save_every_epoch", default=1, type=int, help="How often to save model each epoch step.")
    parser.add_argument("--save_every_step", default=None, type=int, help="How often to save model each step.")
    
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--finetune_type", type=str, help="How to finetune the model.")
    parser.add_argument("--model_type", type=str, help="Type of the model to load.")
    parser.add_argument("--config_name", type=str, help="Where to load the configuration file for the model.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to locally trained model")

    parser.add_argument("--resume_from_checkpoint",type=str2bool, help="Path to locally trained model")
     
    parser.add_argument("--train_file", type=str, default="./data/adult/train.csv", help="A csv file containing the training data.")
    parser.add_argument("--validation_file", type=str, help="A csv file containing the validation data.")
    parser.add_argument("--max_train_samples", default=None, type=int, help="Total number of samples to train with.")
    parser.add_argument("--max_validation_samples", default=None, type=int, help="Total number of samples to evaluate.")
    
    parser.add_argument("--train_batch_size", type=int, default=100, help="Batch size for train dataloader")
    parser.add_argument("--validation_batch_size", type=int, default=8, help="Batch size for evaluation dataloader")
    
    
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Where to store the pretrained models downloaded from huggingface.co.")
    parser.add_argument("--tokenizer_name", type=str, default=None,  help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--use_slow_tokenizer", type=str2bool, default=False, help="If passed, will use a slow tokenizer but not backed the huggingface Tokenizer library.")
    parser.add_argument("--trust_remote_code", type=str2bool, default=False, help=("Whether or not to allow for custom models defined on the Hub in their own modeling files. This option",
                                                                                   "should only be set to `True` for repository you trust and in which you have read the code, as it will "
                                                                                   "execute code present on the Hub in your local machine."))
    parser.add_argument("--low_cpu_mem_usage", type=str2bool, default=False, help=("It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weight "
                                                                                   "are loaded. If passed, LLM loading time and RAM consumption will be benefited."))
    parser.add_argument("--loading_4_bit", type=str2bool, default=False, help="load model in 4 bit for faster training.")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup"
    ])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    
    # DP hyperparams
    parser.add_argument("--enable_privacy", type=str2bool, default=False, help="Turn on privacy mode.")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Batch size for BatchMemoryManager dataloader")
    parser.add_argument("--target_epsilon", type=float, default=1, help="The privacy budget to spend.")
    parser.add_argument("--noise_multiplier", type=float, default=None, help="The scale of noise that will be added to conduct DP-SGD")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="The strength of clipping to constrain the contribution of per-sample gradients in DP-SGD")
    parser.add_argument("--target_delta", type=float, default=1e-5, help="The probability of information accidentally being leaked.")
    
    # Generation
    parser.add_argument("--n_synth_set", type=int, default=1, help="Number of synthetic sets to generate")
    parser.add_argument("--n_synth_samples", type=int, default=None, help="Number of synthetic samples to generate")
    parser.add_argument("--sample_batch", type=int, default=100, help="Sampling batch size")
    parser.add_argument("--temperature", type=float, default=0.7, help="Controls randomness in generated")
    parser.add_argument("--top_p", type=float, default=1.0, help="Controls randomness in generated")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--start_col", type=str, default="income", help="Conditional column")
    parser.add_argument("--prompt_template", type=str, default=None, help="prompt template")
    parser.add_argument("--synth_folder", type=str, default=None, help="folder to store synthetic data.")
    parser.add_argument("--start_prompt", type=str, default="default", help="string or list of string to condition output or specify how to start")
    parser.add_argument("--rejection_sample", type=str2bool, default=False, help="Turn on rejection sampling mode.")
    parser.add_argument("--do_impute", type=str2bool, default=False, help="Turn on imputation mode.")
    parser.add_argument("--sampling_max_retries", type=int, default=15, help="Maximum sampling retries before breaking out of the loop.")
    parser.add_argument("--synth_save_as", type=str, default="synth", help="name to save the synthetic data as.")
    parser.add_argument("--sampling_max_allowed_time", type=int, default=None, help="Maximum sampling time that sampling can take before breaking out of the loop.")
    parser.add_argument("--generation_seed", type=int, default=1000, help="For reproducibility across different generated data.")
    
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True, help="Turn on shuffling mode.")


    # LORA param
    parser.add_argument("--lora_rank", type=int, default=16, help="rank for lora")

    args = parser.parse_args()
    # fmt: on
    return args


def timeit(func):
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        time_elapsed = end_time - start_time
        if not os.path.exists(f"{OUTPUT_DIR}/elapsed_time.txt"):
            with open(f"{OUTPUT_DIR}/elapsed_time.txt", "w") as f:
                f.write("modes,secs,mins,hrs\n")
        with open(f"{OUTPUT_DIR}/elapsed_time.txt", "a") as f:
            f.write(f"{MODES},{time_elapsed},{time_elapsed/60},{time_elapsed/3600}\n")

    return wrapper


@timeit
def main():
    args = parse_args()
    set_seed(args.seed)

    mkdir(args.output_dir)

    logger = get_logger(
        filename=args.output_dir + "/log.log",
    )

    args.finetune_type = (
        None
        if args.finetune_type == "None" or args.finetune_type is None
        else args.finetune_type
    )
    args.model_name_or_path = (
        None
        if args.model_name_or_path == "None" or args.model_name_or_path is None
        else args.model_name_or_path
    )
    args.model_type = (
        None
        if args.model_type == "None" or args.model_type is None
        else args.model_type
    )
    args.config_name = (
        None
        if args.config_name == "None" or args.config_name is None
        else args.config_name
    )
    args.tokenizer_name = (
        None
        if args.tokenizer_name == "None" or args.tokenizer_name is None
        else args.tokenizer_name
    )

    args.max_train_steps = (
        None
        if args.max_train_steps == 0 or args.max_train_steps is None
        else args.max_train_steps
    )

    args.save_every_epoch = (
        None
        if args.save_every_epoch == 0 or args.save_every_epoch is None
        else args.save_every_epoch
    )

    args.save_every_step = (
        None
        if args.save_every_step == 0 or args.save_every_step is None
        else args.save_every_step
    )

    args.weighted_loss = (
        None
        if args.weighted_loss == -1 or args.weighted_loss is None
        else args.weighted_loss
    )

    args.resume_checkpoint_path = (
        None
        if not args.resume_from_checkpoint
        else find_latest_checkpoint(args.output_dir)
    )

    assert not (args.model_name_or_path is None and args.model_type is None)

    global PEFT_MODELS
    PEFT_MODELS = [
        "lora",
        "lokr",
        "layernorm",
        "adalora",
        "gaulora",
        "vera",
        "dora",
        "rslora",
        "rsplora",
        "pissalora",
    ]

    global OUTPUT_DIR
    global MODES
    OUTPUT_DIR = args.output_dir
    MODES = "/".join(
        sum(
            [
                ["train"] if args.train_mode else [],
                ["eval"] if args.evaluation_mode else [],
                ["sample"] if args.generation_mode else [],
            ],
            [],
        )
    )

    logger.info("ARGS")
    logger.info(args)
    logger.info(args.cache_dir)

    # args.resume_checkpoint_path=f"{args.output_dir}/epoch3/resume_checkpoint_dict.pt"

    if args.train_mode:
        # mode = "train"
        dumpy_config_to_json(f"{args.output_dir}/config.json", vars(args))

    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name is not None
        else args.model_name_or_path
    )
    tokenizer = get_tokenizer(
        tokenizer_name,
        cache_dir=args.cache_dir,
        use_slow_tokenizer=args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset, validation_dataset = get_dataset(
        args,
        tokenizer=tokenizer,
    )

    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    if validation_dataset is not None and args.max_validation_samples is not None:
        max_validation_samples = min(
            len(validation_dataset), args.max_validation_samples
        )
        validation_dataset = validation_dataset.select(range(max_validation_samples))

    if validation_dataset is not None:
        logging.info(
            f"Train size={len(train_dataset)}, Validation size={len(validation_dataset)}"
        )
    else:
        logging.info(f"Train size={len(train_dataset)}")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        example = train_dataset[index]
        logger.info(f"Shuffle dataset =  {args.shuffle_dataset}")
        logger.info(f"Sample {index} of the training set: {example}")
        decoded_example = train_dataset.tokenizer.decode(example["input_ids"])
        logger.info(decoded_example)

    #  DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=TGCollator(tokenizer),
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    # print(len(train_dataloader))
    if validation_dataset is not None:
        validation_epoch_dataloader = DataLoader(
            validation_dataset,
            collate_fn=TGCollator(tokenizer),
            batch_size=args.validation_batch_size,
            shuffle=False,
        )
        validation_step_dataloader = DataLoader(
            validation_dataset.select(range(3)),
            collate_fn=TGCollator(tokenizer),
            batch_size=3,
            shuffle=False,
        )
    else:
        validation_step_dataloader = None
        validation_epoch_dataloader = None

    args.train_batch_size = (
        args.train_batch_size
        if len(train_dataset) > args.train_batch_size
        else len(train_dataset)
    )

    model = get_model(
        args.model_name_or_path,
        finetune_type=args.finetune_type,
        model_type=args.model_type,
        config_name=args.config_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        checkpoint_path=args.checkpoint_path,
        logger=logger,
        loading_4_bit=args.loading_4_bit,
        tokenizer=train_dataset.tokenizer,
        lora_rank=args.lora_rank,
    )

    #  Move model to device
    logger.info(model)
    if not args.loading_4_bit:
        model = model.to(args.device)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test. -- currently not used in this project.
    # tokenizer = (
    #     train_dataset.tokenizer
    # )  # in case tokenizer in dataset has been modified
    # embedding_size = model.get_input_embeddings().weight.shape[0]

    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(
        f"Train epoch={args.num_train_epochs}, Max train step={args.max_train_steps}"
    )

    print(args)

    logger.info("**** Statistics ****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num epochs = {args.num_train_epochs}")
    logger.info(f"Train Batch size = {args.train_batch_size}")
    logger.info(f"Total optimization steps = {args.max_train_steps}")
    logger.info("*** ***")

    if args.train_mode:
        model.train()

        progress_bar = tqdm(range(args.max_train_steps))
        completed_steps = 0
        starting_epochs = 1

        # Resume Checkpoint?

        if args.resume_checkpoint_path is not None:
            (
                model,
                optimizer,
                lr_scheduler,
                starting_epochs,
                completed_steps,
            ) = resume_from_checkpoint_files(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_path=args.resume_checkpoint_path,
            )

        if args.enable_privacy:
            privacy_engine = PrivacyEngine(accountant="rdp")
            if args.noise_multiplier is not None:
                model, optimizer, train_dataloader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_dataloader,
                    max_grad_norm=args.max_grad_norm,
                    noise_multiplier=args.noise_multiplier,
                )
                logger.info("*** Privacy Info ***")
                logger.info(f"Noise multiplier (given) = {args.noise_multiplier}")
                logger.info(
                    f"Noise multiplier (computed) = {optimizer.noise_multiplier}"
                )
                logger.info(
                    f"Sigma = {optimizer.noise_multiplier * args.max_grad_norm}"
                )
                logger.info(f"C = {args.max_grad_norm}")
                logger.info(f"Delta = {args.target_delta}")
                logger.info(f"Micro batch size = {args.micro_batch_size}")
                logger.info("*** ***")

            else:
                (
                    model,
                    optimizer,
                    train_dataloader,
                ) = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_dataloader,
                    max_grad_norm=args.max_grad_norm,
                    epochs=args.num_train_epochs,
                    target_epsilon=args.target_epsilon,
                    target_delta=args.target_delta,
                )
                logger.info("*** Privacy Info ***")
                logger.info(f"Noise multiplier = {optimizer.noise_multiplier}")
                logger.info(
                    f"Sigma = {optimizer.noise_multiplier * args.max_grad_norm}"
                )
                logger.info(f"C = {args.max_grad_norm}")
                logger.info(f"Eps = {args.target_epsilon}")
                logger.info(f"Delta = {args.target_delta}")
                logger.info(f"Micro batch size = {args.micro_batch_size}")
                logger.info("*** ***")

            # epsilon, best_alpha = privacy_engine.get_epsilon(args.target_delta)
            # epsilon = privacy_engine.get_epsilon(DELTA)

        progress_bar.update(completed_steps)

        if args.resume_checkpoint_path is not None:
            epoch_logger = EXPLogger(
                args.output_dir + "/perp_per_epoch.txt",
                title=args.finetune_type,
                resume=True,
            )
            step_logger = EXPLogger(
                args.output_dir + "/perp_per_step.txt",
                title=args.finetune_type,
                resume=True,
            )
        else:
            epoch_logger = EXPLogger(
                args.output_dir + "/perp_per_epoch.txt",
                title=args.finetune_type,
                resume=False,
            )
            step_logger = EXPLogger(
                args.output_dir + "/perp_per_step.txt",
                title=args.finetune_type,
                resume=False,
            )
            epoch_logger.set_names(
                [
                    "Epoch",
                    "Epsilon",
                    "Train Perp",
                    "Perp",
                    "Key Perp",
                    "Val Perp",
                    "Other Perp",
                ]
            )
            step_logger.set_names(
                [
                    "Step",
                    "Epsilon",
                    "Train Perp",
                    "Perp",
                    "Key Perp",
                    "Val Perp",
                    "Other Perp",
                ]
            )

        # _data_cnt = 0
        if starting_epochs == args.num_train_epochs + 1:
            raise Exception("All done Champ!")

        for epoch in range(starting_epochs, args.num_train_epochs + 1):
            logger.info("***********")
            logger.info(f"{epoch}, {completed_steps}")
            # _batch_cnt = 0

            if args.enable_privacy:
                if args.micro_batch_size > 0:
                    logger.info("Training with MicroBatch")
                    with BatchMemoryManager(
                        data_loader=train_dataloader,
                        max_physical_batch_size=args.micro_batch_size,
                        optimizer=optimizer,
                    ) as new_train_dataloader:
                        for step, batch in enumerate(new_train_dataloader):
                            assert len(batch["input_ids"]) <= args.micro_batch_size

                            # Remove token_type_ids if present
                            if "token_type_ids" in batch:
                                del batch["token_type_ids"]

                            # move data to device
                            batch = {i: v.to(args.device) for i, v in batch.items()}

                            outputs = model(**batch)
                            loss = outputs.loss

                            if args.weighted_loss is not None:
                                loss_2 = compute_disentangled_loss_for_training(
                                    outputs,
                                    batch,
                                    train_dataset,
                                    alpha=args.weighted_loss,
                                )
                                loss = loss_2

                            loss.backward()
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()

                            try:
                                perplexity = math.exp(loss.detach())
                            except OverflowError:
                                perplexity = float("inf")

                            if not optimizer._is_last_step_skipped:
                                # _batch_cnt = 0
                                completed_steps += 1

                                privacy_spent_so_far = privacy_engine.get_epsilon(
                                    args.target_delta
                                )

                                if (
                                    args.save_every_step is not None
                                    and completed_steps % args.save_every_step == 0
                                ):
                                    # tmp
                                    args.current_epoch = epoch
                                    args.current_step = completed_steps
                                    args.lr_scheduler = lr_scheduler
                                    args.optimizer = optimizer
                                    if epoch > 1:
                                        continue
                                    elif completed_steps in [25, 50, 75, 100]:
                                        savedir = (
                                            args.output_dir + f"/step{completed_steps}"
                                        )
                                        model_to_save = model._module
                                        save_model(model_to_save, savedir, args)
                                    else:
                                        savedir = (
                                            args.output_dir + f"/step{completed_steps}"
                                        )
                                        model_to_save = model._module
                                        save_model(model_to_save, savedir, args)

                                step_or_epoch_logging_and_plotting(
                                    args,
                                    logger,
                                    model,
                                    perplexity,
                                    privacy_spent_so_far,
                                    validation_dataset,
                                    validation_step_dataloader,
                                    step_logger,
                                    completed_steps,
                                    epoch,
                                    "Step",
                                )

                                progress_bar.update(1)

                                if completed_steps >= args.max_train_steps:
                                    break

                else:
                    for step, batch in enumerate(train_dataloader):
                        # Remove token_type_ids if present
                        if "token_type_ids" in batch:
                            del batch["token_type_ids"]

                        # move data to device
                        batch = {i: v.to(args.device) for i, v in batch.items()}

                        outputs = model(**batch)
                        loss = outputs.loss

                        if args.weighted_loss is not None:
                            loss_2 = compute_disentangled_loss_for_training(
                                outputs, batch, train_dataset, alpha=args.weighted_loss
                            )
                            loss = loss_2

                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        # import pdb; pdb.set_trace()

                        try:
                            perplexity = math.exp(loss.detach())
                        except OverflowError:
                            perplexity = float("inf")

                        completed_steps += 1

                        privacy_spent_so_far = privacy_engine.get_epsilon(
                            args.target_delta
                        )

                        if (
                            args.save_every_step is not None
                            and completed_steps % args.save_every_step == 0
                        ):
                            # tmp
                            args.current_epoch = epoch
                            args.current_step = completed_steps
                            args.lr_scheduler = lr_scheduler
                            args.optimizer = optimizer
                            if epoch > 1:
                                continue
                            elif completed_steps in [25, 50, 75, 100]:
                                savedir = args.output_dir + f"/step{completed_steps}"
                                model_to_save = model._module
                                save_model(model_to_save, savedir, args)
                            else:
                                savedir = args.output_dir + f"/step{completed_steps}"
                                model_to_save = model._module
                                save_model(model_to_save, savedir, args)

                        step_or_epoch_logging_and_plotting(
                            args,
                            logger,
                            model,
                            perplexity,
                            privacy_spent_so_far,
                            validation_dataset,
                            validation_step_dataloader,
                            step_logger,
                            completed_steps,
                            epoch,
                            "Step",
                        )

                        progress_bar.update(1)

                        if completed_steps >= args.max_train_steps:
                            break
            else:
                for step, batch in enumerate(train_dataloader):
                    privacy_spent_so_far = "None"

                    # Remove token_type_ids if present
                    if "token_type_ids" in batch:
                        del batch["token_type_ids"]

                    # move data to device
                    batch = {i: v.to(args.device) for i, v in batch.items()}

                    outputs = model(**batch)
                    loss = outputs.loss

                    if args.weighted_loss is not None:
                        loss_2 = compute_disentangled_loss_for_training(
                            outputs, batch, train_dataset, alpha=args.weighted_loss
                        )
                        loss = loss_2

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    try:
                        perplexity = math.exp(loss.detach())
                    except OverflowError:
                        perplexity = float("inf")

                    completed_steps += 1

                    if (
                        args.save_every_step is not None
                        and completed_steps % args.save_every_step == 0
                    ):
                        # tmp
                        args.current_epoch = epoch
                        args.current_step = completed_steps
                        args.lr_scheduler = lr_scheduler
                        args.optimizer = optimizer
                        if epoch > 1:
                            continue
                        elif completed_steps in [25, 50, 75, 100]:
                            savedir = args.output_dir + f"/step{completed_steps}"
                            model_to_save = model
                            save_model(model_to_save, savedir, args)
                        else:
                            savedir = args.output_dir + f"/step{completed_steps}"
                            model_to_save = model
                            save_model(model_to_save, savedir, args)

                    step_or_epoch_logging_and_plotting(
                        args,
                        logger,
                        model,
                        perplexity,
                        privacy_spent_so_far,
                        validation_dataset,
                        validation_step_dataloader,
                        step_logger,
                        completed_steps,
                        epoch,
                        "Step",
                    )

                    progress_bar.update(1)

                    if completed_steps >= args.max_train_steps:
                        break

            ###################End of Epoch############
            step_or_epoch_logging_and_plotting(
                args,
                logger,
                model,
                perplexity,
                privacy_spent_so_far,
                validation_dataset,
                validation_epoch_dataloader,
                epoch_logger,
                completed_steps,
                epoch,
                "Epoch",
            )

            savedir = args.output_dir + f"/epoch{epoch}"
            if args.save_every_epoch is not None and epoch % args.save_every_epoch == 0:
                if args.enable_privacy:
                    model_to_save = model._module
                else:
                    model_to_save = model
                args.current_epoch = epoch
                args.current_step = completed_steps
                args.lr_scheduler = lr_scheduler
                args.optimizer = optimizer
                save_model(model_to_save, savedir, args)

        if args.enable_privacy:
            model_to_save = model._module
        else:
            model_to_save = model

        args.current_epoch = epoch
        args.current_step = completed_steps
        args.lr_scheduler = lr_scheduler
        args.optimizer = optimizer

        save_model(model_to_save, args.output_dir, args)

    if args.evaluation_mode:
        print("PERP EVALUATION MODE")
        print(args.output_dir)
        assert validation_dataset is not None
        key_perp, val_perp, other_perp, total_perp = compute_disentangled_loss(
            model=model,
            dataloader=validation_epoch_dataloader,
            dataset=validation_dataset,
            device=args.device,
        )
        logger.info(
            f"perp = {total_perp}, key_perp = {key_perp}, val_perp = {val_perp}, other_perp = {other_perp}"
        )
        name = (
            args.checkpoint_path
            if args.checkpoint_path is not None
            else args.output_dir
        )
        if not os.path.exists(args.output_dir + "/eval.txt"):
            with open(args.output_dir + "/eval.txt", "w") as f:
                f.write("name,size,total perp,key perp,value perp,other perp\n")

        with open(args.output_dir + "/eval.txt", "a") as f:
            f.write(
                f"{name},{len(validation_dataset)},{total_perp},{key_perp},{val_perp},{other_perp}\n"
            )

    if args.generation_mode:
        print("GENERATION MODE")
        from generation import generate

        n_samples = (
            args.n_synth_samples
            if args.n_synth_samples is not None
            else len(train_dataset)
        )

        if args.n_synth_set > 1:
            for i in range(args.n_synth_set):
                sample = generate(
                    n_samples=n_samples,
                    model=model,
                    dataset=train_dataset,
                    start_prompt=args.start_prompt,
                    start_col=args.start_col,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    k=args.sample_batch,
                    max_length=args.max_new_tokens,
                    drop_nan=args.rejection_sample,
                    do_impute=args.do_impute,
                    prompt_template=args.prompt_template,
                    device=args.device,
                    max_retries=args.sampling_max_retries,
                    max_allowed_time=args.sampling_max_allowed_time,
                    save_folder=args.synth_folder
                    if args.synth_folder is not None
                    else args.output_dir,
                    save_name=f"{args.synth_save_as}_{i}",
                    seed=args.generation_seed * (i + 1),
                )
        else:
            sample = generate(
                n_samples=n_samples,
                model=model,
                dataset=train_dataset,
                start_prompt=args.start_prompt,
                start_col=args.start_col,
                temperature=args.temperature,
                top_p=args.top_p,
                k=args.sample_batch,
                max_length=args.max_new_tokens,
                drop_nan=args.rejection_sample,
                do_impute=args.do_impute,
                prompt_template=args.prompt_template,
                device=args.device,
                max_retries=args.sampling_max_retries,
                max_allowed_time=args.sampling_max_allowed_time,
                save_folder=args.synth_folder
                if args.synth_folder is not None
                else args.output_dir,
                save_name=f"{args.synth_save_as}",
                seed=args.generation_seed,
            )


def step_or_epoch_logging_and_plotting(
    args,
    logger,
    model,
    train_perplexity,
    privacy_spent_so_far,
    validation_dataset,
    validation_dataloader,
    step_or_epoch_logger,
    completed_steps,
    epoch,
    logging_mode,
):
    # during training, I want to test on very small validation set at every step
    if validation_dataloader is not None:
        key_perp, val_perp, other_perp, total_perp = compute_disentangled_loss(
            model=model,
            dataloader=validation_dataloader,
            dataset=validation_dataset,
            device=args.device,
        )
        logger.info(
            f"epoch={epoch}/{args.num_train_epochs}, step = {completed_steps}/{args.max_train_steps}, epsilon={privacy_spent_so_far}, Train perp = {train_perplexity}, perp = {total_perp}, key_perp = {key_perp}, val_perp = {val_perp}, other_perp = {other_perp}"
        )

    else:
        logger.info(
            f"epoch={epoch}/{args.num_train_epochs}, step = {completed_steps}/{args.max_train_steps}, epsilon={privacy_spent_so_far}, Train perp = {train_perplexity}"
        )
        key_perp, val_perp, other_perp, total_perp = "None", "None", "None", "None"

    if logging_mode == "Epoch":
        step_or_epoch_logger.append(
            [
                epoch,
                privacy_spent_so_far,
                train_perplexity,
                total_perp,
                key_perp,
                val_perp,
                other_perp,
            ]
        )
    else:
        step_or_epoch_logger.append(
            [
                completed_steps,
                privacy_spent_so_far,
                train_perplexity,
                total_perp,
                key_perp,
                val_perp,
                other_perp,
            ]
        )
    step_or_epoch_logger.plot(
        ["Perp", "Key Perp", "Val Perp", "Other Perp"], x=f"{logging_mode}"
    )
    savefig(os.path.join(args.output_dir, f"perp_{logging_mode.lower()}.png"))

    step_or_epoch_logger.plot(["Train Perp"], x=f"{logging_mode}")
    savefig(os.path.join(args.output_dir, f"train_perp_{logging_mode.lower()}.png"))


def get_logger(filename=None):
    logger = logging.getLogger(__name__)
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%m/%d/%Y %I:%M:%S %p"
    encoding = "utf-8"

    formatter = logging.Formatter(fmt=format, datefmt=datefmt)
    logger.setLevel(logging.DEBUG)

    if filename is not None:
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(filename, encoding=encoding)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_model(model, savedir, args):
    mkdir(savedir)
    model.config.save_pretrained(savedir)
    model.generation_config.save_pretrained(savedir)
    safetensors.torch.save_model(
        model, savedir + "/model.safetensors", metadata={"format": "pt"}
    )

    if args.finetune_type in PEFT_MODELS:
        model.peft_config["default"].save_pretrained(savedir)
        model.save_pretrained(savedir, safe_serialization=True)

    # Save the optimizer, scheduler, and training state for resuming later
    # Save the random seed states
    rng_states = {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": args.optimizer.state_dict(),
        "scheduler_state_dict": args.lr_scheduler.state_dict(),
        "epoch": args.current_epoch,
        "step": args.current_step,
        "rng_states": rng_states,
    }
    checkpoint_path = os.path.join(savedir, f"resume_checkpoint_dict.pt")
    torch.save(checkpoint, checkpoint_path)

    print(f"Model and checkpoint saved at {savedir}")


def get_dataset(args, tokenizer):
    train_file = args.train_file
    validation_file = args.validation_file
    serializer = args.serializer
    train_df = pd.read_csv(train_file)

    if not args.shuffle_dataset:
        column_names = [args.start_col] + sorted(
            list(set(train_df) - set([args.start_col]))
        )
        train_df = train_df[column_names]

    if validation_file is not None:
        validation_df = pd.read_csv(validation_file)
        if not args.shuffle_dataset:
            validation_df = validation_df[column_names]

        validation_dataset = LLMtgDataset.from_pandas(validation_df)
        validation_dataset.set_serializer(serializer)
        validation_dataset.set_tokenizer(tokenizer)
        validation_dataset.set_shuffler(shuffle=args.shuffle_dataset)

        train_dataset = LLMtgDataset.from_pandas(train_df)
        train_dataset.set_serializer(serializer)
        train_dataset.set_tokenizer(tokenizer)
        train_dataset.set_shuffler(shuffle=args.shuffle_dataset)

    else:
        train_dataset = LLMtgDataset.from_pandas(train_df)
        train_dataset.set_serializer(serializer)
        train_dataset.set_tokenizer(tokenizer)

        train_dataset.set_shuffler(shuffle=args.shuffle_dataset)

        validation_dataset = None

    return train_dataset, validation_dataset


def get_tokenizer(
    tokenizer_name, cache_dir=None, use_slow_tokenizer=False, trust_remote_code=False
):
    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "use_fast": not use_slow_tokenizer,
        "trust_remote_code": trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_config(
    config_name,
    from_pretrained=True,
    cache_dir=None,
    trust_remote_code=False,
    logger=None,
):
    if logger is None:
        logger = get_logger()

    config_kwargs = {"cache_dir": cache_dir, "trust_remote_code": trust_remote_code}
    if not from_pretrained:
        logger.info("Instantiating config from scratch.")
        config = CONFIG_MAPPING[config_name]()
    else:
        config = AutoConfig.from_pretrained(config_name, **config_kwargs)

    return config


def get_model(
    model_name_or_path,
    finetune_type=None,
    model_type=None,
    config_name=None,
    checkpoint_path=None,
    cache_dir=None,
    trust_remote_code=False,
    low_cpu_mem_usage=False,
    loading_4_bit=False,
    logger=None,
    tokenizer=None,
    lora_rank=16,
):
    if logger is None:
        logger = get_logger()

    if checkpoint_path == "None":
        checkpoint_path = None

    if loading_4_bit:
        bnb_config_4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        low_cpu_mem_usage = True
    else:
        bnb_config_4 = None

    if model_name_or_path is None or finetune_type is None:
        config = get_config(
            config_name=model_type,
            from_pretrained=False,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            logger=logger,
        )
    else:
        config_name = config_name if config_name is not None else model_name_or_path
        config = get_config(
            config_name=config_name,
            from_pretrained=True,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            logger=logger,
        )

    if model_name_or_path is None or finetune_type is None:
        logger.info("Instantiating model from scratch.")
        model = AutoModelForCausalLM.from_config(
            config=config, trust_remote_code=trust_remote_code
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
            quantization_config=bnb_config_4,
            cache_dir=cache_dir,
        )

    if finetune_type != "entire":
        for p in model.parameters():
            p.requires_grad = False

        if finetune_type in PEFT_MODELS:
            model = get_peft_model_wrapper(
                model,
                model_type=config.model_type,
                adapter=finetune_type,
                rank=lora_rank,
            )
        elif finetune_type in ["resb4", "append"]:
            customized_head = AugmentedBlock(finetune_type, model.lm_head)
            model.lm_head = customized_head

    if tokenizer is not None:
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            logger.info(
                f"Resizing model token embeddings size ({embedding_size}) to match tokenizer length ({len(tokenizer)})"
            )
            model.resize_token_embeddings(len(tokenizer))

    model = load_model_from_checkpoint(model, checkpoint_path)
    n_params = total_trainable_params(model)
    logger.info(f"Finetuning {finetune_type} - Total size={n_params/2**20:.2f}M params")
    return model


def total_trainable_params(model):
    n_params = sum(
        {
            p.data_ptr(): p.numel() for p in model.parameters() if p.requires_grad
        }.values()
    )
    return n_params


def load_model_from_checkpoint(model, checkpoint_path, logger=None):
    if logger is None:
        logger = get_logger()

    if checkpoint_path is not None:
        if "safetensors" in checkpoint_path:
            ckpt = {}
            with safe_open(checkpoint_path, framework="pt") as f:
                for k in f.keys():
                    ckpt[k] = f.get_tensor(k)
        else:
            ckpt = torch.load(checkpoint_path)

        mismatch_keys = set(model.state_dict().keys()) - set(ckpt.keys())
        logger.info(f"Mismatched keys: {mismatch_keys}")

        model.load_state_dict(ckpt, strict=False)
        logger.info("Checkpoint successfully loaded.")
    return model


def resume_from_checkpoint_files(model, optimizer, lr_scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    #  Restore RNG states
    torch.set_rng_state(checkpoint["rng_states"]["torch_rng_state"])
    if (
        torch.cuda.is_available()
        and checkpoint["rng_states"]["cuda_rng_state"] is not None
    ):
        torch.cuda.set_rng_state_all(checkpoint["rng_states"]["cuda_rng_state"])
    np.random.set_state(checkpoint["rng_states"]["numpy_rng_state"])
    random.setstate(checkpoint["rng_states"]["python_rng_state"])

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    return model, optimizer, lr_scheduler, epoch + 1, step


def get_peft_model_wrapper(model, model_type, adapter="lora", rank=16):
    if adapter == "lokr" and model_type == "gpt2":
        raise NotImplementedError(
            f"Adapter={adapter} doesn't currently support Conv1d layers."
        )

    if model_type == "gpt2":
        target_modules = ["c_attn", "c_fc", "c_proj"]
        task_type = TaskType.CAUSAL_LM

    if model_type == "llama":
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        # target_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
        task_type = TaskType.CAUSAL_LM

    if adapter == "lora":
        config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=task_type,
            bias="none",
            target_modules=target_modules,
        )
    elif adapter == "dora":
        config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=task_type,
            bias="none",
            target_modules=target_modules,
            use_dora=True,
        )
    elif adapter == "pissalora":
        config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=task_type,
            bias="none",
            target_modules=target_modules,
            init_lora_weights="pissa",
        )
    elif adapter == "rslora":
        config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=task_type,
            bias="none",
            target_modules=target_modules,
            use_rslora=True,
        )
    elif adapter == "rsplora":
        config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=task_type,
            bias="none",
            target_modules=target_modules,
            init_lora_weights="pissa",
            use_rslora=True,
        )
    elif adapter == "adalora":
        config = AdaLoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=task_type,
            bias="none",
            target_modules=target_modules,
        )
    elif adapter == "gaulora":
        config = LoraConfig(
            r=rank,
            lora_alpha=16,
            lora_dropout=0.0,
            task_type=task_type,
            bias="none",
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
    elif adapter == "vera":
        config = VeraConfig(
            r=rank,
            vera_dropout=0.0,
            task_type=task_type,
        )
    elif adapter == "lokr":
        config = LoKrConfig(
            r=rank,
            target_modules=target_modules,
            alpha=16,
            rank_dropout=0.0,
            module_dropout=0.0,
        )
    elif adapter == "layernorm":
        config = LNTuningConfig(task_type=task_type)

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


@dataclass
class TGCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


@torch.no_grad()
def compute_disentangled_loss(model, dataloader, dataset, device):
    model.eval()
    disentangled_losses = [[], [], [], []]
    losses = []

    save_full_path = f"/eval_perp_full-shuffle={dataset.shuffle}.txt"
    if not os.path.exists(OUTPUT_DIR + save_full_path):
        with open(OUTPUT_DIR + save_full_path, "w") as f:
            f.write("step, total perp,key perp,value perp,other perp\n")

    key_val_sep_id = dataset.others_token_id["key_val_sep"]
    text_sep_id = dataset.others_token_id["text_sep"]
    prefix_id = dataset.others_token_id["prefix"]
    prefix_id = prefix_id if prefix_id is not None else []
    eos_token_id = [dataset.tokenizer.eos_token_id]
    bos_token_id = [dataset.tokenizer.bos_token_id]

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            # Remove token_type_ids if present
            if "token_type_ids" in batch:
                del batch["token_type_ids"]

            # move data to device
            batch = {i: v.to(device) for i, v in batch.items()}

            outputs = model(**batch)

        logits = outputs.logits  # (batch_size, token_len, vocal_size)
        labels = batch["labels"]  # (batch_size, token_len)

        key_masks, val_masks, other_masks = [], [], []
        for sq in labels:
            km, vm, om = [False] * len(sq), [False] * len(sq), [False] * len(sq)
            is_key = True

            for i_t, t in enumerate(sq):
                if (
                    t
                    in prefix_id
                    + key_val_sep_id
                    + text_sep_id
                    + eos_token_id
                    + bos_token_id
                ):
                    om[i_t] = True
                    if dataset.serializer == "apval":
                        if t in key_val_sep_id:
                            is_key = not is_key
                    else:
                        if t in key_val_sep_id + text_sep_id:
                            is_key = not is_key
                elif is_key:
                    km[i_t] = True
                else:
                    vm[i_t] = True

            key_masks.append(km)
            val_masks.append(vm)
            other_masks.append(om)

        key_masks = torch.tensor(key_masks).to(logits.device)
        # print(labels[0])
        # print(labels[0][key_masks[0]])
        # print(labels[0][val_masks[0]])
        # print(labels[0][other_masks[0]])

        key_masks = (
            key_masks.view(-1, key_masks.size(-1))[..., 1:].contiguous().view(-1)
        )

        val_masks = torch.tensor(val_masks).to(logits.device)
        val_masks = (
            val_masks.view(-1, val_masks.size(-1))[..., 1:].contiguous().view(-1)
        )

        other_masks = torch.tensor(other_masks).to(logits.device)
        other_masks = (
            other_masks.view(-1, other_masks.size(-1))[..., 1:].contiguous().view(-1)
        )

        # print("shape of key mask", key_masks.shape)

        # Shift so that token < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        unfold_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        disentangled_losses[0].append(unfold_loss[key_masks])
        disentangled_losses[1].append(unfold_loss[val_masks])
        disentangled_losses[2].append(unfold_loss[other_masks])
        disentangled_losses[3].append(unfold_loss)

        with open(OUTPUT_DIR + save_full_path, "a") as f:
            f.write(
                f"{step}, {torch.mean(unfold_loss)},{torch.mean(unfold_loss[key_masks])},{torch.mean(unfold_loss[val_masks])},{torch.mean(unfold_loss[other_masks])}\n"
            )

        loss = outputs.loss
        losses.append(loss.repeat(len(batch["labels"])))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    for i in range(4):
        disentangled_losses[i] = torch.cat(disentangled_losses[i])

    # default value in case of exploding values
    disentangled_metrics = [float("inf")] * 4

    for i in range(4):
        try:
            disentangled_metrics[i] = math.exp(torch.mean(disentangled_losses[i]))
        except OverflowError:
            pass

    model.train()
    return disentangled_metrics


def compute_disentangled_loss_for_training(outputs, batch, dataset, alpha=1 / 3):
    disentangled_losses = [[], [], [], []]
    losses = []

    key_val_sep_id = dataset.others_token_id["key_val_sep"]
    text_sep_id = dataset.others_token_id["text_sep"]
    prefix_id = dataset.others_token_id["prefix"]
    prefix_id = prefix_id if prefix_id is not None else []
    eos_token_id = [dataset.tokenizer.eos_token_id]
    bos_token_id = [dataset.tokenizer.bos_token_id]

    logits = outputs.logits  # (batch_size, token_len, vocal_size)
    labels = batch["labels"]  # (batch_size, token_len)

    key_masks, val_masks, other_masks = [], [], []
    for sq in labels:
        km, vm, om = [False] * len(sq), [False] * len(sq), [False] * len(sq)
        is_key = True

        for i_t, t in enumerate(sq):
            if (
                t
                in prefix_id
                + key_val_sep_id
                + text_sep_id
                + eos_token_id
                + bos_token_id
            ):
                om[i_t] = True
                if dataset.serializer == "apval":
                    if t in key_val_sep_id:
                        is_key = not is_key
                else:
                    if t in key_val_sep_id + text_sep_id:
                        is_key = not is_key
            elif is_key:
                km[i_t] = True
            else:
                vm[i_t] = True

        key_masks.append(km)
        val_masks.append(vm)
        other_masks.append(om)

    # import pdb; pdb.set_trace()
    key_masks = torch.tensor(key_masks).to(logits.device)

    key_masks = key_masks.view(-1, key_masks.size(-1))[..., 1:].contiguous().view(-1)

    val_masks = torch.tensor(val_masks).to(logits.device)
    val_masks = val_masks.view(-1, val_masks.size(-1))[..., 1:].contiguous().view(-1)

    other_masks = torch.tensor(other_masks).to(logits.device)
    other_masks = (
        other_masks.view(-1, other_masks.size(-1))[..., 1:].contiguous().view(-1)
    )

    # Shift so that token < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    unfold_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )

    disentangled_losses[0].append(unfold_loss[key_masks])
    disentangled_losses[1].append(unfold_loss[val_masks])
    disentangled_losses[2].append(unfold_loss[other_masks])
    disentangled_losses[3].append(unfold_loss)

    for i in range(4):
        disentangled_losses[i] = torch.cat(disentangled_losses[i]).mean()

    # print(disentangled_losses[0], disentangled_losses[1], disentangled_losses[2], disentangled_losses[3])

    loss = outputs.loss
    losses.append(loss.repeat(len(batch["labels"])))
    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)

    value_loss = disentangled_losses[1]
    other_loss = disentangled_losses[2]
    key_loss = disentangled_losses[0]

    format_loss = key_loss + other_loss

    # loss = 0.5 * format_loss + 0.5 * value_loss

    # loss = ((0.5*(1-alpha))  * key_loss) +  ((0.5*(1-alpha))  * other_loss) + (alpha * value_loss)

    loss = ((1 - alpha) * format_loss) + (alpha * value_loss)

    return loss


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory by looking for files
    that match the naming pattern of a checkpoint (e.g., epoch numbers).
    """
    checkpoint_files = []

    # Walk through the directory and find all checkpoint files
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.startswith("resume_checkpoint_dict"):
                fullpath = os.path.join(root, file)
                if "/epoch" in fullpath or "/step" in fullpath:
                    checkpoint_files.append(os.path.join(root, file))

    print(checkpoint_files)
    if not checkpoint_files:
        return None

    # Sort the checkpoint files by epoch number (assuming the format contains epoch numbers)
    # Extract epoch number from file names assuming format includes 'epoch<number>'

    try:
        checkpoint_files.sort(key=lambda x: int(x.split("/epoch")[-1].split("/")[0]))
    except:
        try:
            checkpoint_files.sort(key=lambda x: int(x.split("/step")[-1].split("/")[0]))
        except:
            return None

    # The last one should be the latest checkpoint
    latest_checkpoint = checkpoint_files[-1]

    return latest_checkpoint


def calc_grad_norm(model, return_counter=False):
    from collections import Counter

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm(2).item()
    return grad_norms if not return_counter else Counter(grad_norms)


if __name__ == "__main__":
    main()
