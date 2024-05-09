from transformers import AutoModelForTokenClassification

models = set()
for s in ["large", "base", "small", "xsmall"]:
    models.add(f"microsoft/deberta-v3-{s}")

for s in ["14m", "70m", "160m", "410m", "1b", "1.4b"]:
    models.add(f"EleutherAI/pythia-{s}")
for s in ["base", "small", "large", "small", "xl"]:
    models.add(f"google/t5-v1_1-{s}")
    models.add(f"google/mt5-{s}")
