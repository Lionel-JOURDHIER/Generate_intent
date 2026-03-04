import ollama
import csv
import json
import os
import re
from collections import defaultdict
# from ressources.seeds import SEEDS
from services.get_seeds import SEEDS

# --- CONFIGURATION ---
MODEL_NAME = "qwen2.5-coder:1.5b"
TARGET_TOTAL = 768
OUTPUT_FILE = "nlp_dataset.csv"

# --- FONCTIONS UTILITAIRES ---

def load_existing_csv(filepath):
    """
    Charge le CSV existant pour reprendre là où on s'est arrêté.
    Détecte automatiquement les noms de colonnes.
    """
    seen_texts = set()
    count_per_intent = defaultdict(int)

    if not os.path.exists(filepath):
        with open(filepath, mode="w", encoding="utf-8", newline='') as f:
            csv.writer(f).writerow(["text", "intent"])
        print(f"[INIT] Nouveau fichier créé : {filepath}")
        return seen_texts, count_per_intent

    with open(filepath, mode="r", encoding="utf-8", newline='') as f:
        reader = csv.DictReader(f)
        
        # Affiche les colonnes trouvées pour debug
        fieldnames = reader.fieldnames or []
        print(f"[INFO] Colonnes détectées : {fieldnames}")

        # Détection flexible des colonnes text et intent
        col_text   = next((c for c in fieldnames if c.strip().lower() in ("text", "texte", "phrase", "utterance")), None)
        col_intent = next((c for c in fieldnames if c.strip().lower() in ("intent", "intention", "label", "class")), None)

        if not col_text or not col_intent:
            # Fallback : prendre les deux premières colonnes
            if len(fieldnames) >= 2:
                col_text, col_intent = fieldnames[0], fieldnames[1]
                print(f"[WARN] Colonnes non reconnues, fallback → text='{col_text}' intent='{col_intent}'")
            else:
                print("[ERREUR] CSV invalide : moins de 2 colonnes détectées.")
                return seen_texts, count_per_intent

        for row in reader:
            text   = row[col_text].strip()
            intent = row[col_intent].strip()
            if text:
                seen_texts.add(text)
                count_per_intent[intent] += 1

    total = sum(count_per_intent.values())
    print(f"[REPRISE] {total} lignes trouvées (col_text='{col_text}', col_intent='{col_intent}')")
    for intent, count in count_per_intent.items():
        print(f"  · {intent}: {count}")

    return seen_texts, count_per_intent


def append_to_csv(filepath, text, intent):
    """Écrit une ligne immédiatement et force le flush disque."""
    with open(filepath, mode="a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([text, intent])
        f.flush()
        os.fsync(f.fileno())  # Garantie d'écriture physique


def generate_variation(text_example, intent_name):
    """Génère une variation via Ollama et retourne une chaîne propre ou None."""

    prompt = f"""Tu es un générateur de données NLU. 
    Génère 1 reformulation de cette phrase pour l'intention "{intent_name}".
    Phrase de base : "{text_example}"
    Réponds UNIQUEMENT avec un objet JSON : {{"variation": "ta phrase ici"}}"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={
                "temperature": 0.8,
                "num_predict": 80,
                "num_ctx": 1024,
            },
        )
        raw = response["message"]["content"].strip()
        data = json.loads(raw)

        # --- Extraction robuste : on cherche la première string non vide ---
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and len(value) > 3:
                    return value.strip()

        if isinstance(data, list) and len(data) > 0:
            return str(data[0]).strip()

    except json.JSONDecodeError:
        # Fallback : extraire le contenu entre guillemets le plus long
        matches = re.findall(r'"([^"]{8,})"', raw)
        if matches:
            return max(matches, key=len).strip()

    except Exception as e:
        print(f"  [ERREUR] {e} | raw={raw[:80]}")

    return None


# --- SETUP ---
num_intents = len(SEEDS)
needed_per_intent = TARGET_TOTAL // num_intents
print(f"Cible : {needed_per_intent} phrases × {num_intents} intentions = ~{TARGET_TOTAL} total\n")

# Chargement de l'état existant (reprise automatique)
seen_texts, count_per_intent = load_existing_csv(OUTPUT_FILE)
total_written = sum(count_per_intent.values())

# --- BOUCLE DE GÉNÉRATION ---
for intent, examples in SEEDS.items():
    already_done = count_per_intent[intent]

    if already_done >= needed_per_intent:
        print(f"[SKIP] {intent} ({already_done}/{needed_per_intent} — complet)")
        continue

    print(f"\n--- {intent} : {already_done}/{needed_per_intent} déjà faits ---")
    count_intent = already_done

    while count_intent < needed_per_intent:
        for seed_text in examples:
            if count_intent >= needed_per_intent:
                break

            variation = generate_variation(seed_text, intent)

            if variation and variation not in seen_texts:
                seen_texts.add(variation)
                append_to_csv(OUTPUT_FILE, variation, intent)

                count_intent += 1
                total_written += 1
                progress = (count_intent / needed_per_intent) * 100
                print(
                    f"  [{total_written:>5}] {intent} {count_intent}/{needed_per_intent}"
                    f" ({progress:.0f}%) → {variation[:70]}"
                )
            else:
                # Doublon ou génération vide : on réessaie silencieusement
                pass

print(f"\n✅ Terminé ! {total_written} lignes dans '{OUTPUT_FILE}'")