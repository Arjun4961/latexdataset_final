import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import TrOCRProcessor
from transformers import convert_slow_tokenizer

from transformers import VisionEncoderDecoderModel

from datasets import load_metric

import os
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import accelerate

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=256):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['Image'][idx]
        text = self.df['Text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# Loading Train and Eval Sets into Environment
train = pd.read_csv(input("Enter the Path for Train Set!!"))
eval = pd.read_csv(input("Enter the Path for Eval Set!!"))
print("Loading Sets into the Env is Completed....")

# Getting Image Files Path
ImgFilePath = input("Enter Path for Image Files!!")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
print("Processor Loaded into Environment Successfully...")

# Converting DataSet into IAM Format
print("Attempting to Convert Sets into IAM Format???")
train_dataset = IAMDataset(root_dir = ImgFilePath, df = train, processor = processor)
eval_dataset = IAMDataset(root_dir = ImgFilePath, df = eval, processor=processor)
print("Format conversion Successfull...")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
print("Model Loaded into Env Successfully...")

cer_metric = load_metric("cer")
print("Metrics Loaded into Env Successfully...")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}
      
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
      
print("Parameters Initialized...")

# Specify the directory path in your Google Drive
drive_output_dir = input("Please Enter OutPut Directory!!")

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    num_train_epochs = 10,  #changed to 5 from 25
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
    output_dir=drive_output_dir,
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
)
print("Training Arguments Created...")

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
print("Defined Trainer Object...")

try:
    # Train the model
    print("Initiating Training Procedure???")
    trainer.train()
except (KeyboardInterrupt, RuntimeError) as e:
    # Handle keyboard interrupt and runtime disconnect errors
    print(f"Error during training: {e}")
finally:
    # Save the model even if there's an interruption or error
    print("Saving model...")
    model.save_pretrained(drive_output_dir)
    #training_args.save_model_args(drive_output_dir)
    print("Model saved in Google Drive.")
