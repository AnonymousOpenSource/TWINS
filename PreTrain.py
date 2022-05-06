import torch
torch.cuda.is_available()
from transformers import RobertaConfig
from pathlib import Path
from tokenizers import CharBPETokenizer
import os
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from Tokenizer.InstructionTokenizer import InstructionTokenizer
from transformers import LineByLineTextDataset
import pickle
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


# Save path
BlockEmbedding_path = "./BlockEmbedding"

# Assembly sequences
corpus_path = "database/corpus.pkl"

config = RobertaConfig(
    vocab_size=32_000,
    max_position_embeddings=512,
    num_attention_heads=8,
    num_hidden_layers=6,
    hidden_size=128,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)
device = torch.device('cuda:0')
model.to(device)

dataDir = "./database/binaries/Train"
InsTokenizer = pickle.load(open("Tokenizer/model_save/tokenizer.model", "rb"))
datasets = []
datasets = pickle.load(open(corpus_path, "rb"))


data_collator = DataCollatorForLanguageModeling(
     tokenizer=InsTokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=BlockEmbedding_path +"/model_store",
    overwrite_output_dir=True,
    num_train_epochs=12,
    per_device_train_batch_size=24,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=1e-4,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=datasets,

)

torch.cuda.empty_cache()
trainer.train()


