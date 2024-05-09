from transformers import AutoModelForTokenClassification

models = set()
for s in ["large", "base", "small", "xsmall"]:
    models.add(f"microsoft/deberta-v3-{s}")

for s in ["14m", "70m", "160m", "410m", "1b", "1.4b"]:
    models.add(f"EleutherAI/pythia-{s}")
for s in ["base", "small", "large", "small"]:
    models.add(f"google/t5-v1_1-{s}")
    models.add(f"google/mt5-{s}")

out = {}
for k in sorted(models):
    print(f"Checking {k}")
    model = AutoModelForTokenClassification.from_pretrained(k, num_labels=9)
    out[k.replace("/", "_")] = model.num_parameters()

import json

print(json.dumps(out, sort_keys=True, indent=2))
