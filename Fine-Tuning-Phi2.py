from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import datasets

conll2003 = datasets.load_dataset("conll2003")

train = conll2003["train"]
print(train[1])
val = conll2003["validation"]
test = conll2003["test"]

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token

with open('prompt-base.txt', 'r') as file:
    prompt_base = file.read()

train_tokens = [x['tokens'] for x in train]
train_label = [x['ner_tags'] for x in train]

val_tokens = [x['tokens'] for x in val]
val_label = [x['ner_tags'] for x in val]

test_tokens = [x['tokens'] for x in test]
test_label = [x['ner_tags'] for x in test]

training_prompts = [prompt_base + ]

training_arguments = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500, 
    logging_steps=10,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)
