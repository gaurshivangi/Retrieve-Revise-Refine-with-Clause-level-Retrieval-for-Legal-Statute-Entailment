import json
import re
from pathlib import Path

def extract_clauses(article_text):
    """Extract individual clauses from article text to support clause-informed prompting."""
    # Remove article number prefix if present
    text = re.sub(r'^Article \d+:?\s*', '', article_text)
    
    # Split into clauses - looking for numbered paragraphs and natural sentence breaks
    clauses = []
    
    # First try to split by numbered paragraphs
    paragraph_splits = re.split(r'\(\d+\)', text)
    if len(paragraph_splits) > 1:
        for i, p in enumerate(paragraph_splits[1:], 1):  # Skip first empty split
            clauses.append(f"({i}){p.strip()}")
    else:
        # If no numbered paragraphs, split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clauses = [s.strip() for s in sentences if s.strip()]
    
    return clauses

def create_civil_code_json(input_path, output_path):
    """Create JSON with article content and extracted clauses for clause-informed prompting."""
    content = Path(input_path).read_text(encoding='utf-8')
    
    # Split by "Article X" headers
    articles = {}
    parts = re.split(r'(Article\s+\d+)', content)
    
    # Process parts in groups of article header + content
    for i in range(1, len(parts), 2):
        art_label = parts[i].strip()
        art_num = re.search(r'\d+', art_label).group()
        art_text = parts[i+1].strip()
        
        # Extract clauses for this article
        full_text = f"{art_label}: {art_text}"
        clauses = extract_clauses(art_text)
        
        articles[art_num] = {
            "content": full_text,
            "clauses": clauses
        }
    
    # Save with pretty formatting
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    print(f"Created {output_path} with {len(articles)} articles")
    print(f"Average clauses per article: {sum(len(art['clauses']) for art in articles.values()) / len(articles):.1f}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "COLIEE2025statute_data-English/text/civil_code_en-1to724-2.txt"
    output_path = base_dir / "data/civil_code_en.json"
    
    create_civil_code_json(input_path, output_path)
