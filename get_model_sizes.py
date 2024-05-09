from transformers import AutoModelForTokenClassification

models = set()
for s in ["large", "base", "small", "xsmall"]:
    models.add(f"microsoft/deberta-v3-{s}")

for s in ["14m", "70m", "160m", "410m", "1b", "1.4b"]:
    models.add(f"EleutherAI/pythia-{s}")
for s in ["base", "small", "large", "small", "xl"]:
    models.add(f"google/t5-v1_1-{s}")
    models.add(f"google/mt5-{s}")

out = {}
for k in models:
    model = AutoModelForTokenClassification.from_pretrained(k, num_labels=9)
    out[k.replace("/", "_")] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

import json

print(json.dumps(out, sort_keys=True, indent=2))
