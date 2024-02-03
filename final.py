import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from datasets import load_metric
from sklearn.model_selection import train_test_split

# Define the Loader function
def Loader(df, maxe):
    ImgSourceDir = '/content/drive/My Drive/ImgToLatex/LatexImages/'
    FileNames = list(df.columns[2:6])
    FinalDF = pd.DataFrame(columns=['Image', 'Text'])
    NotFoundFiles = []
    count = 0

    for i in range(len(df)):
        DirectoryName = df.iloc[i]['DirectoryName']
        for FileDpi in FileNames:
            try:
                text = df.iloc[i]['Latex Code']
                ImgFile = ImgSourceDir + DirectoryName + '/' + df.iloc[i][FileDpi] + '.png'
                img = Image.open(ImgFile)

                newRow = pd.DataFrame({"Image": [ImgFile], "Text": [text]})
                FinalDF = pd.concat([FinalDF, newRow], ignore_index=True)
                count += 1
                print(len(FinalDF))
                if count == maxe:
                    return (FinalDF, NotFoundFiles)
            except FileNotFoundError:
                NotFoundFiles.append(df.iloc[i][FileDpi])
    return (FinalDF, NotFoundFiles)

# Read Excel files and split into train and test sets
SourceExcelFile = pd.read_excel(Path)
train_df, test_df = train_test_split(SourceExcelFile, test_size=0.3)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Load data using Loader function
train, files = Loader(train_df, 10000)
train.to_csv('/content/drive/My Drive/ImgToLatex/First10000/train.csv')

eval, efiles = Loader(test_df, 2000)
eval.to_csv('/content/drive/My Drive/ImgToLatex/First10000/eval.csv')

# Define Dataset class
class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=256):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['Image'][idx]
        text = self.df['Text'][idx]
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# Load TrOCRProcessor and create datasets
ImgFilePath = '/content/drive/MyDrive/LatexDataset_02_01_2024/LatexImages'
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir=ImgFilePath, df=train, processor=processor)
eval_dataset = IAMDataset(root_dir=ImgFilePath, df=eval, processor=processor)

# Initialize model
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# Load CER metric
cer_metric = load_metric("cer")

# Define compute_metrics function
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# Specify training arguments
drive_output_dir = "/content/drive/MyDrive/ImgToLatex/TrOcr"
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=25,
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

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

# Train the model
try:
    trainer.train()
except (KeyboardInterrupt, RuntimeError) as e:
    print(f"Error during training: {e}")
finally:
    print("Saving model...")
    model.save_pretrained(drive_output_dir)
    print("Model saved in Google Drive.")

# Test the trained model
TestPath = '/content/drive/My Drive/ImgToLatex/LatexImages/1-1500/latex_image_750_dpi_200_normal.png'
image = Image.open(TestPath).convert("RGB")
Model = VisionEncoderDecoderModel.from_pretrained(drive_output_dir)
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = Model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
