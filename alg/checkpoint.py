import os

import torch
from accelerate import Accelerator


def _gather_lora_state_dict_zero3(accelerator: Accelerator, student_model):
    zero_stage = 0
    if hasattr(accelerator, "deepspeed_engine_wrapped"):
        engine = accelerator.deepspeed_engine_wrapped.engine
        zero_stage = engine.zero_optimization_stage()

    if zero_stage == 0:
        model = accelerator.unwrap_model(student_model)
        lora_sd = {}
        if accelerator.is_main_process:
            for name, p in model.named_parameters():
                if ("lora_" in name) or ("modules_to_save" in name):
                    lora_sd[name] = p.detach().cpu().clone()
        return lora_sd

    if zero_stage in [1, 2]:
        model = accelerator.unwrap_model(student_model)
        lora_sd = {}
        if accelerator.is_main_process:
            for name, p in model.named_parameters():
                if ("lora_" in name) or ("modules_to_save" in name):
                    lora_sd[name] = p.detach().cpu().clone()
        return lora_sd

    from deepspeed import zero

    model = accelerator.unwrap_model(student_model)
    lora_sd = {}
    names = []
    params = []
    for name, p in model.named_parameters():
        if ("lora_" in name) or ("modules_to_save" in name):
            names.append(name)
            params.append(p)

    for name, p in zip(names, params):
        with zero.GatheredParameters([p], modifier_rank=0):
            if accelerator.is_main_process:
                lora_sd[name] = p.detach().cpu().clone()

    return lora_sd


def save_checkpoint_lora_zero3(
    accelerator: Accelerator,
    student_model,
    tokenizer,
    meta_net,
    save_path: str,
):
    accelerator.wait_for_everyone()
    lora_sd = _gather_lora_state_dict_zero3(accelerator, student_model)

    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        if len(lora_sd) == 0:
            raise RuntimeError("lora_sd is empty; LoRA may not be injected.")

        unwrapped_student = accelerator.unwrap_model(student_model)
        unwrapped_student.save_pretrained(
            save_path,
            state_dict=lora_sd,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(save_path)
        if meta_net is not None:
            torch.save(
                accelerator.unwrap_model(meta_net).state_dict(),
                os.path.join(save_path, "meta_net.pt"),
            )

    accelerator.wait_for_everyone()
