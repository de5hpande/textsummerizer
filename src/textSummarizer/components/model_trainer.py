from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from src.textSummarizer.entity import ModelTrainerConfig
import torch
from datasets import load_from_disk
import os
os.environ["WANDB_DISABLED"] = "true"

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config=config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        model_Falcons = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt,
        torch_dtype=torch.float16,quantization_config={"load_in_4bit": True}).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_Falcons)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # For text summarization models
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none"
        )

        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        #loading the data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16,
             report_to= "none"
        ) 
        trainer = Trainer(model=model_Falcons, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"],
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_Falcons.save_pretrained(os.path.join(self.config.root_dir,"Falcons-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))