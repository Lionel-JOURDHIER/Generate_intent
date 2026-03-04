import csv
import os
import sys

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE = "dataset.csv"  # ← changez si besoin
OUTPUT_FILE = "dataset_cleaned.csv"  # fichier de sortie

KEYWORDS = [
    "en tant qu'ia",
    "en tant qu'ia",  # variante avec apostrophe typographique
    "intent",
    "désolé",
    "desole",  # variante sans accent
]
# ───────────────────────────────────────────────────────────────


def contains_keyword(text: str) -> bool:
    """Retourne True si le texte contient l'un des mots-clés (insensible à la casse)."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in KEYWORDS)


def clean_csv(input_path: str, output_path: str) -> dict:
    stats = {"total": 0, "removed": 0, "kept": 0}

    with open(input_path, encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader, None)
        rows = list(reader)

    stats["total"] = len(rows)
    kept_rows = []

    for row in rows:
        # Vérifie toutes les colonnes de la ligne
        if any(contains_keyword(cell) for cell in row):
            stats["removed"] += 1
        else:
            kept_rows.append(row)

    stats["kept"] = len(kept_rows)

    with open(output_path, encoding="utf-8", newline="") as check:
        pass  # juste pour vérifier qu'on peut écrire

    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        if header:
            writer.writerow(header)
        writer.writerows(kept_rows)

    return stats


if __name__ == "__main__":
    # Permet de passer le fichier en argument : python clean_intents.py mon_fichier.csv
    input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE

    if not os.path.exists(input_file):
        print(f"[ERREUR] Fichier introuvable : {input_file}")
        sys.exit(1)

    print(f"Traitement de : {input_file}")
    stats = clean_csv(input_file, output_file)

    print("\n✅ Terminé !")
    print(f"   Total lignes  : {stats['total']:,}")
    print(f"   Supprimées    : {stats['removed']:,}")
    print(f"   Conservées    : {stats['kept']:,}")
    print(f"   Fichier propre: {output_file}")
