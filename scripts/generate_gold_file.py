import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path to find the module
sys.path.append(str(Path(__file__).parent.parent / 'Retrieve-Revise-Refine-master'))

from extract_gold_task3_task4 import load_samples_from_file

def generate_gold_file(input_dir, output_file):
    all_samples = {}
    for filename in os.listdir(input_dir):
        if filename.startswith('riteval_R') and filename.endswith('.xml'):
            xml_path = os.path.join(input_dir, filename)
            print(f"Processing {xml_path}...")
            samples = load_samples_from_file(xml_path)
            all_samples.update(samples)
    
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"Successfully created {output_file} with {len(all_samples)} samples.")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "COLIEE2025statute_data-English/train"
    output_file = base_dir / "data/gold_R_series.json"
    generate_gold_file(input_dir, output_file)
