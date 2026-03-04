import csv


def load_seeds(path = "dataset_cleaned.csv"):
    """

    """
    dico = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = row["intent"]
            if intent not in dico:
                dico[intent] = []
            dico[intent].append(row["phrase"])
    return dict(dico)

SEEDS = load_seeds()
