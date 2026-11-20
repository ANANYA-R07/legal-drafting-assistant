import csv
from pathlib import Path

path = Path("data/legal_corpus.csv")

rows = []
with path.open("r", encoding="utf-8-sig") as f:
    lines = f.readlines()

for line_number, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue

    if line_number == 0:
        rows.append(["clause_type", "sentence"])
        continue

    if "," not in line:
        print(f"Skipped malformed line: {line_number + 1}")
        continue

    category, sentence = line.split(",", 1)
    rows.append([category.strip(), sentence.strip()])

with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)

print("CSV repair completed.")
print("Total data rows:", len(rows) - 1)
