import os

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from checkpoint import save_checkpoint_lora_zero3
from config import ACGPFConfig
from data import MultiTeacherCoTDataset, acgpf_collate_fn
from trainer import ACGPFTrainer


def main():
    config = ACGPFConfig()
    deepspeed_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_clipping=1.0,
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin, mixed_precision="bf16"
    )

    if accelerator.is_main_process:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    if config.use_lora:
        model.config.use_cache = False
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    dataset = MultiTeacherCoTDataset(config.dataset_path, tokenizer)
    collate = lambda x: acgpf_collate_fn(x, tokenizer)

    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    trainer_module = ACGPFTrainer(model, dataset.thinking_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trainer_module, optimizer, dataloader = accelerator.prepare(
        trainer_module, optimizer, dataloader
    )

    trainer_module.train()
    for epoch in range(config.epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            with accelerator.accumulate(trainer_module):
                loss = trainer_module(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["gold_input_ids"],
                    batch["assistant_start_positions"],
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainer_module.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if (epoch + 1) % config.save_epochs == 0:
            save_path = os.path.join(
                config.checkpoint_dir, f"checkpoint-epoch-{epoch}"
            )
            if config.use_lora:
                save_checkpoint_lora_zero3(
                    accelerator, trainer_module.model, tokenizer, None, save_path
                )
            else:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.makedirs(save_path, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(trainer_module.model)
                    unwrapped.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
