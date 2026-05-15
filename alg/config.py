import os


class ACGPFConfig:
    model_name = os.getenv(
        "MODEL_NAME",
        "/data/gjq/my_project/multi-view-cot-distillation/model/qwen2_5-1_5b",
    )
    dataset_path = os.getenv(
        "DATASET_PATH",
        "/data/gjq/my_project/multi-teacher-cot-distillation/dataset/math_data/sampled_200.json",
    )
    output_path = os.getenv("OUTPUT_PATH", "./output/qwen2_5-1_5b")

    use_lora = True
    lora_r = int(os.getenv("LORA_R", "16"))
    lora_alpha = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout = float(os.getenv("LORA_DROPOUT", "0.1"))
    lora_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    max_seq_len = int(os.getenv("MAX_SEQ_LEN", "2560"))
    learning_rate = float(os.getenv("LEARNING_RATE", "1e-5"))
    beta_1 = float(os.getenv("BETA_1", "1.0"))
    beta_2 = float(os.getenv("BETA_2", "1.0"))
    beta_3 = float(os.getenv("BETA_3", "0.5"))
    temperature = float(os.getenv("TEMPERATURE", "1.0"))

    batch_size = int(os.getenv("BATCH_SIZE", "1"))
    gradient_accumulation_steps = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
    epochs = int(os.getenv("EPOCHS", "5"))

    lambda_mcc = float(os.getenv("LAMBDA_MCC", "0.1"))

    thinking_keywords = [
        "Therefore",
        "Thus",
        "Hence",
        "So",
        "Consequently",
        "Wait",
        "But",
        "However",
        "Although",
        "First",
        "Second",
        "Finally",
        "Specifically",
        "In summary",
        "Basically",
        "Let",
        "Assuming",
        "Suppose",
        "Hmm",
        "Okay",
        "Note",
    ]

    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    save_epochs = int(os.getenv("SAVE_EPOCHS", "1"))
