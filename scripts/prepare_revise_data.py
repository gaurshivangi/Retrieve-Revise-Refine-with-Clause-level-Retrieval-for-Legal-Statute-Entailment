"""
Script to prepare data files needed for Revise stage:
1. Extract gold data from XML files
2. Extract civil code from text file
"""

import json
import xml.etree.ElementTree as ET
import os
import re
from pathlib import Path


def extract_gold_from_xml(xml_file_path, output_json_path):
    """
    Extract gold data from COLIEE XML file format.
    
    XML format (Task 3):
    <ROOT>
      <pair id="R02-1" label="Y">
        <t1>Article 886 ... Article 887 ...</t1>
        <t2>Query text...</t2>
      </pair>
      ...
    </ROOT>
    """
    import re
    
    def _parse_article_fix(article_text):
        """Extract article IDs from text"""
        return re.findall(r'(?<=^Article )([^ \n]+)', article_text, re.MULTILINE)
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    gold_data = {}
    
    for i in range(len(root)):
        pair = root[i]
        query = None
        rel_article_ids = []
        
        for elem in pair:
            if elem.tag == "t1":
                # Extract article IDs from t1 text
                if elem.text:
                    rel_article_ids = _parse_article_fix(elem.text.strip())
            elif elem.tag == "t2":
                # Query text
                if elem.text:
                    query = elem.text.strip()
        
        if query is not None:
            sample_id = pair.attrib.get('id', f"sample_{i}")
            gold_data[sample_id] = {
                'label': pair.attrib.get('label', 'N'),
                'rel_article_ids': rel_article_ids,
                'query': query
            }
        else:
            print(f"[Warning] Sample {pair.attrib.get('id', i)} is ignored (no query text)")
    
    # Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(gold_data)} queries from {xml_file_path}")
    print(f"Saved to {output_json_path}")
    
    return gold_data


def extract_civil_code_from_text(text_file_path, output_json_path):
    """
    Extract civil code articles from text file.
    
    Expected format:
    Article 1  (1) Text...
    Article 2  Text...
    Article 3-2  Text...
    ...
    """
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    civil_code = {}
    current_article_id = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match "Article NUMBER" or "Article NUMBER-NUMBER"
        article_match = re.match(r'^Article\s+([\d-]+)\s+(.*)$', line)
        if article_match:
            # Save previous article if exists
            if current_article_id is not None:
                article_text = '\n'.join(current_content).strip()
                if article_text:
                    civil_code[current_article_id] = {
                        "content": f"Article {current_article_id}  {article_text}"
                    }
            
            # Start new article
            current_article_id = article_match.group(1)
            remaining_text = article_match.group(2).strip()
            current_content = [remaining_text] if remaining_text else []
        else:
            # Continue current article
            if current_article_id is not None:
                current_content.append(line)
    
    # Save last article
    if current_article_id is not None:
        article_text = '\n'.join(current_content).strip()
        if article_text:
            civil_code[current_article_id] = {
                "content": f"Article {current_article_id}  {article_text}"
            }
    
    # Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(civil_code, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {len(civil_code)} articles from {text_file_path}")
    print(f"Saved to {output_json_path}")
    
    return civil_code


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data files for Revise stage")
    parser.add_argument("--xml_file", type=str, 
                       default="../COLIEE2025statute_data-English/train/riteval_R02_en.xml",
                       help="Path to XML file with gold data")
    parser.add_argument("--text_file", type=str,
                       default="../COLIEE2025statute_data-English/text/civil_code_en-1to724-2.txt",
                       help="Path to text file with civil code")
    parser.add_argument("--output_dir", type=str,
                       default="./Retrieve-Revise-Refine-master/data",
                       help="Output directory for data files")
    parser.add_argument("--gold_output", type=str,
                       default="gold_task3_task4.json",
                       help="Output filename for gold data")
    parser.add_argument("--civil_code_output", type=str,
                       default="civil_code_en.json",
                       help="Output filename for civil code")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract gold data
    gold_output_path = os.path.join(args.output_dir, args.gold_output)
    if os.path.exists(args.xml_file):
        extract_gold_from_xml(args.xml_file, gold_output_path)
    else:
        print(f"Warning: XML file not found: {args.xml_file}")
        print("Please provide the correct path to your XML file")
    
    # Extract civil code
    civil_code_output_path = os.path.join(args.output_dir, args.civil_code_output)
    if os.path.exists(args.text_file):
        extract_civil_code_from_text(args.text_file, civil_code_output_path)
    else:
        print(f"Warning: Text file not found: {args.text_file}")
        print("Please provide the correct path to your civil code text file")
    
    print("\nData preparation complete!")
    print(f"Gold data: {gold_output_path}")
    print(f"Civil code: {civil_code_output_path}")


if __name__ == "__main__":
    main()

