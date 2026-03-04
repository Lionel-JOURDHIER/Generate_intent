"""
Générateur de phrases d'entraînement NLP par intention via Ollama.
Sortie : CSV écrit ligne par ligne en temps réel.

Usage : python generate_intent_phrases.py --intents intents.txt --output dataset.csv
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:1.5b"  # Changez selon votre modèle Ollama installé
PHRASES_PER_INTENT = 40  # Nombre de phrases à générer par intention
BATCH_SIZE = 10  # Phrases générées par appel LLM
TEMPERATURE = 0.9  # Créativité
LANGUAGE = "français"  # Langue cible


# ──────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────
def build_prompt(intent: str, batch_size: int, recent: list[str]) -> str:
    avoid_section = ""
    if recent:
        avoid_section = (
            "\nÉvite de répéter ces formulations déjà générées :\n"
            + "\n".join(f"- {p}" for p in recent)
        )

    return f"""Tu es un expert en génération de données d'entraînement pour la classification d'intentions NLP.

Génère exactement {batch_size} phrases différentes en {LANGUAGE} qui expriment l'intention : "{intent}"

Consignes :
- Varie les formulations : questions, affirmations, formules courtes et longues
- Varie le registre : formel, familier, direct, indirect
- Chaque phrase doit être naturelle, comme écrite par un utilisateur réel
- Pas de numérotation, pas de tirets, une phrase par ligne
- Ne génère rien d'autre que les phrases{avoid_section}

Phrases :"""


# ──────────────────────────────────────────────
# OLLAMA STREAMING
# ──────────────────────────────────────────────
def call_ollama_stream(prompt: str):
    """
    Appel Ollama en mode stream.
    Yield les lignes complètes au fur et à mesure de leur réception.
    """
    global MODEL, TEMPARATURE
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 2048,
        },
    }

    buffer = ""
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except Exception:
                continue

            buffer += chunk.get("response", "")

            # Yield chaque ligne complète dès qu'elle est disponible
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield line

            if chunk.get("done") and buffer.strip():
                yield buffer  # Flush du dernier fragment
                break


# ──────────────────────────────────────────────
# NETTOYAGE
# ──────────────────────────────────────────────
def clean_line(line: str) -> str | None:
    cleaned = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
    return cleaned if len(cleaned) > 5 else None


# ──────────────────────────────────────────────
# GENERATION + ECRITURE CSV EN TEMPS REEL
# ──────────────────────────────────────────────
def generate_and_write(
    intent: str, target: int, writer: csv.writer, csvfile, seen: set[str]
) -> int:
    """
    Génère les phrases pour une intention et les écrit dans le CSV au fil de l'eau.
    Retourne le nombre de phrases effectivement écrites.
    """
    global BATCH_SIZE
    count = 0
    attempts = 0
    max_attempts = (target // BATCH_SIZE) + 5
    history: list[str] = []

    print(f"\n  🎯 Intention : '{intent}'")

    while count < target and attempts < max_attempts:
        batch = min(BATCH_SIZE, target - count)
        print(f"     Batch {attempts + 1} → demande {batch} phrases ({count}/{target})")

        try:
            prompt = build_prompt(intent, batch, history[-10:])
            batch_count = 0

            for raw_line in call_ollama_stream(prompt):
                phrase = clean_line(raw_line)
                if not phrase:
                    continue

                phrase_lower = phrase.lower()
                if phrase_lower in seen:
                    continue

                # ── Écriture immédiate dans le CSV ──
                seen.add(phrase_lower)
                history.append(phrase)
                writer.writerow([intent, phrase])
                csvfile.flush()  # Force l'écriture disque token par token

                print(f"       ✍️  [{count + 1}] {phrase}", flush=True)
                count += 1
                batch_count += 1

                if count >= target:
                    break

            print(f"     ✅ +{batch_count} phrases (total : {count})")

        except requests.exceptions.RequestException as e:
            print(f"     ❌ Erreur Ollama : {e} — nouvelle tentative dans 3s")
            time.sleep(3)

        attempts += 1
        time.sleep(0.3)

    return count


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    global MODEL, PHRASES_PER_INTENT
    parser = argparse.ArgumentParser(
        description="Génère des phrases NLP par intention via Ollama — sortie CSV temps réel"
    )
    parser.add_argument(
        "--intents", required=True, help="Fichier texte avec une intention par ligne"
    )
    parser.add_argument("--output", default="dataset.csv", help="Fichier CSV de sortie")
    parser.add_argument(
        "--count", type=int, default=PHRASES_PER_INTENT, help="Phrases par intention"
    )
    parser.add_argument("--model", default=MODEL, help="Modèle Ollama à utiliser")
    args = parser.parse_args()

    MODEL = args.model
    PHRASES_PER_INTENT = args.count

    intents_path = Path(args.intents)
    if not intents_path.exists():
        print(f"❌ Fichier introuvable : {intents_path}")
        return

    intents = [
        l.strip()
        for l in intents_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"📋 {len(intents)} intention(s) : {intents}")
    print(f"🤖 Modèle  : {MODEL} | {PHRASES_PER_INTENT} phrases/intention")
    print(f"💾 Sortie  : {args.output}\n")

    total = 0

    # Le fichier reste ouvert pendant toute la génération
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["intent", "phrase"])  # En-tête
        csvfile.flush()

        for intent in intents:
            seen: set[str] = set()
            n = generate_and_write(intent, PHRASES_PER_INTENT, writer, csvfile, seen)
            total += n
            print(f"  → {n} phrases écrites pour '{intent}'")

    print(f"\n✅ Terminé ! {total} phrases écrites dans : {args.output}")


if __name__ == "__main__":
    main()
