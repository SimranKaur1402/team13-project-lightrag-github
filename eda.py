import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import requests

# 1. Configuration
data_sources = {
    "Deep_Learning_Book": "https://aikosh.indiaai.gov.in/static/Deep+Learning+Ian+Goodfellow.pdf",
    "WHO_Healthcare": "https://applications.emro.who.int/dsaf/dsa664.pdf"
}

def analyze_pdf(name, url):
    print(f"--- Processing {name} ---")
    try:
        response = requests.get(url, timeout=20)
        doc = fitz.open(stream=response.content, filetype="pdf")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return None

    text_per_page = []
    symbol_counts = 0
    full_text = ""

    for page in doc:
        # Try standard extraction first
        page_text = page.get_text("text")
        
        # If standard extraction yields nothing, try "blocks" mode
        # This extracts text based on visual positioning which helps with odd encodings
        if len(page_text.strip()) < 10:
            blocks = page.get_text("blocks")
            page_text = " ".join([b[4] for b in blocks if isinstance(b[4], str)])

        full_text += page_text
        text_per_page.append(len(page_text.split()))
        
        # Count math/tech symbols (Common in Deep Learning Book)
        symbol_counts += len(re.findall(r'[α-ωΑ-Ωθ∂∑∫√π±≠≈≤≥]', page_text))

    # Tokenizing
    words = re.findall(r'\w+', full_text.lower())
    unique_words = set(words)
    lexical_diversity = (len(unique_words) / len(words)) * 100 if words else 0
    
    # Filter keywords (words > 5 chars to avoid "the", "and", etc.)
    keywords = [w for w in words if len(w) > 5]
    top_keywords = Counter(keywords).most_common(10)
    
    return {
        "name": name,
        "pages": len(doc),
        "word_counts": text_per_page,
        "lexical_diversity": lexical_diversity,
        "symbols": symbol_counts,
        "top_keywords": top_keywords,
        "total_words": len(words)
    }

# 2. Run Analysis
results = [analyze_pdf(k, v) for k, v in data_sources.items()]
results = [r for r in results if r is not None] # Filter failures

# 3. Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, res in enumerate(results):
    sns.histplot(res['word_counts'], ax=axes[i], kde=True, 
                 color='skyblue' if i==0 else 'salmon', bins=30)
    axes[i].set_title(f"Word Density: {res['name']}")
    axes[i].set_xlabel("Words per Page")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# 4. Final Evidence Table
df_evidence = pd.DataFrame(results).drop(columns=['word_counts'])
print("\n--- FINAL EDA EVIDENCE TABLE ---")
print(df_evidence.to_string())
