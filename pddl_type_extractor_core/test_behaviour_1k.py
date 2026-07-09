import csv
import json
from pathlib import Path

from pddl_type_extractor_core.pipeline import derive_types_for_instruction

INPUT_CSV = Path("activity_names_with_instructions.csv")
OUTPUT_CSV = Path("activity_names_with_types.csv")


def process_behaviour_1k(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV):
    with input_csv.open("r", newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))

    fieldnames = list(rows[0].keys()) + [
        "types",
        "propbank_frames_considered",
        "propbank_role_descriptions_considered",
        "verbnet_classes_considered",
    ]

    total = len(rows)
    print(f"\nProcessing {total} instructions")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print("=" * 80)

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            instruction = row.get("instruction", "")
            print(f"\n[{i}/{total}] {instruction}")

            try:
                result = derive_types_for_instruction(instruction)

                row["types"] = json.dumps(result["types"])
                row["propbank_frames_considered"] = json.dumps(result["propbank_frames_considered"])
                row["propbank_role_descriptions_considered"] = json.dumps(result["propbank_role_descriptions_considered"])
                row["verbnet_classes_considered"] = json.dumps(result["verbnet_classes_considered"])

                print(f"  PropBank frames: {result['propbank_frames_considered'] or 'NONE'}")
                print(f"  VerbNet classes: {result['verbnet_classes_considered'] or 'NONE'}")
                print(f"  Derived types:   {result['types'] or 'NONE'}")

            except Exception as e:
                row["types"] = json.dumps([])
                row["propbank_frames_considered"] = json.dumps([])
                row["propbank_role_descriptions_considered"] = json.dumps([])
                row["verbnet_classes_considered"] = json.dumps([])

                print("  ERROR")
                print(f"  {e}")

            writer.writerow(row)

    print("\nDone.")
    print(f"Saved output to: {output_csv}")


def main():
    process_behaviour_1k()


if __name__ == "__main__":
    main()
