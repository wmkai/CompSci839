from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import datasets
import evaluate
import random
import ast

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

outputs = []
for test_prompt in test_prompts:
    output = generator(test_prompt, max_new_tokens = 300)
    output = output[0]['generated_text'][len(test_prompt):]
    output = output[output.index('['):]
    if output.find(']') != -1:
        output = output[:output.find(']')+1]
    else:
        output = output + ']'
    output = ast.literal_eval(output)
    output = [x.strip() for x in output]
    print(output)
    outputs.append(output)

metric = evaluate.load_metric("seqeval")
results = metric.compute(predictions=outputs, references=test_label)
print("Precision: " + results["overall_precision"])
print("Recall: " + results["overall_recall"])
print("f1: " + results["overall_f1"])
print("Accuracy: " + results["overall_accuracy"])