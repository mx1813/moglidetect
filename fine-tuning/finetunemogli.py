import pandas as pd
import numpy as np
import evaluate
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from datasets import DatasetDict

"""Im ersten Schritt wird das Datenset, welches die Trainingsdaten enthält, geladen. Der Datenatz umfasst Texte und ordnet diesem einem Label zu (0=menschlich; 1=KI). In diesem Fall liegen die Daten in einer csv-Datei vor. Die Datei wird mithilfe von pandas eingelesen."""

data = pd.read_csv('dsFinale.csv', on_bad_lines='skip', sep=";", encoding="utf-8")
print(data.head())

"""Der Datensatz werden in 3 Datensätze aufgeteilt: Train, Test und Validate. Anschließend werden die datensätze für das Training vorbereitet."""

validate, train, test  = \
            np.split(data.sample(frac=1, random_state=42),
                    [int(.1*len(data)), int(.8*len(data))])


# data preparation
## reset indices
train = train.reset_index()[['text','label']]
test = test.reset_index()[['text','label']]
validate = validate.reset_index()[['text','label']]
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

"""Anschließend werden die Datensätze tokenisiert, damit das vortrainierte Modell diese verwenden kann."""

tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_ds = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
id2label = {0: "HUMAN", 1: "AI"}
label2id = {"HUMAN": 0, "AI": 1}

"""Metriken für die Bewertung der Modellleistung während des Trainings initialisieren"""

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

"""Modell und Trainigsparameter auswählen"""

model = AutoModelForSequenceClassification.from_pretrained('deepset/gbert-base', num_labels=2, id2label=id2label, label2id=label2id) # Pre-trained model

training_args = TrainingArguments(
    output_dir="mogli02",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False
)

"""Trainer API initialisieren und mit entsprechenden Parametern ausführen"""

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

"""Nach dem Training das Modell nochmal in Ordner abspeichern"""

trainer.save_model("mogli02_sec")