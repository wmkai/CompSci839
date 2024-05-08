from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import datasets
import random

conll2003 = datasets.load_dataset("conll2003")

train = conll2003["train"]
val = conll2003["validation"]
test = conll2003["test"]

train_tokens = [x['tokens'] for x in train]
train_label = [x['ner_tags'] for x in train]

test_tokens = [x['tokens'] for x in test]
test_label = [x['ner_tags'] for x in test]

with open('prompt-base.txt', 'r') as file:
    prompt_base = file.read()

few_shot_pool = ["\nInput: " + str(train_tokens[i]) + "\nOutput: " + str(train_label) for i in range(len(train_tokens))]

num_examples = 0

random.seed(1234)

test_prompts = []
for i in range(len(test_tokens)):
    test_prompt = prompt_base
    few_shot_ex = random.sample(few_shot_pool, num_examples)
    for j in range(num_examples):
        test_prompt += few_shot_ex[j]
    test_prompt += "\nInput: " + str(test_tokens[i]) + "\nOutput: "
    test_prompts.append(test_prompt)

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
model.to("cuda:0")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

print(generator(test_prompts[0], max_new_tokens = 300))