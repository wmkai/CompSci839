from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import datasets

conll2003 = datasets.load_dataset("conll2003")

train = conll2003["train"]
val = conll2003["validation"]
test = conll2003["test"]

with open('prompt-base.txt', 'r') as file:
    prompt_base = file.read()

train_tokens = [x['tokens'] for x in train]
train_label = [x['ner_tags'] for x in train]

val_tokens = [x['tokens'] for x in val]
val_label = [x['ner_tags'] for x in val]

test_tokens = [x['tokens'] for x in test]
test_label = [x['ner_tags'] for x in test]

train_prompts = [prompt_base + "\nInput: " + str(tokens) + "\nOutput: " for tokens in train_tokens]
train_output = [str(label) for label in train_label]
test_prompts = [prompt_base + "\nInput: " + str(tokens) + "\nOutput: " for tokens in test_tokens]
test_output = [str(label) for label in test_label]

train_dataset = [{"prompt": train_prompts[i], "completion": train_output[i]} for i in range(len(train_prompts))]

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    optim="adamw",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    lr_scheduler_type="constant"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=500,
    tokenizer=tokenizer,
    args=training_arguments
)
