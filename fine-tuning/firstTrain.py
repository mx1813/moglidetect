import pandas as pd
import numpy as np
import evaluate
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset
from datasets import DatasetDict

data = pd.read_csv('dsFinale.csv', on_bad_lines='skip', sep=";", encoding="utf-8")
print(data.head())

validate, train, test  = \
            np.split(data.sample(frac=1, random_state=42), 
                    [int(.1*len(data)), int(.8*len(data))])


# data preparation
## reset indices
train = train.reset_index()[['text','labels']]
test = test.reset_index()[['text','labels']]
validate = validate.reset_index()[['text','labels']]
## dataframes to datadict
tds = Dataset.from_pandas(train)
testds = Dataset.from_pandas(test)
vds = Dataset.from_pandas(validate)
tds.to_csv("dsTrain.csv", sep=";", index=False, encoding="utf-8")
testds.to_csv("dsTest.csv", sep=";", index=False, encoding="utf-8")
vds.to_csv("dsValidate.csv", sep=";", index=False, encoding="utf-8")
dataset = DatasetDict()
dataset['train'] = tds
dataset['test'] = testds
dataset['validate'] = vds
print(dataset)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_ds = dataset.map(preprocess_function, batched=True)
batch_size = 32
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
id2label = {0: "HUMAN", 1: "AI"}
label2id = {"HUMAN": 0, "AI": 1}

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

bert_model = AutoModelForSequenceClassification.from_pretrained('dbmdz/bert-base-german-cased', num_labels=2, id2label=id2label, label2id=label2id) # Pre-trained model

training_args = TrainingArguments(
    output_dir="my_first_model",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
print("Done.")
trainer.save_model("sec_model")