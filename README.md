Project Overview
This project implements a complete machine learning pipeline to analyze web content for SEO quality assessment and near-duplicate detection. The system processes raw content, extracts NLP features, scores content quality with an ML model, and flags 'thin' content.




🔗 Deployed Streamlit URL (Bonus +15 Points)
The final interactive demonstration of the pipeline is deployed here (required for bonus points ):


Deployed Application: [PASTE YOUR DEPLOYED STREAMLIT URL HERE]

🛠️ Setup and Installation
This project requires Python 3.9+ and dependencies listed in requirements.txt.

Clone the Repository:

Bash

git clone https://github.com/saranya-myrepository/seo-content-detector.git
cd seo-content-detector
Install Dependencies:

Bash

pip install -r requirements.txt
Download NLTK Data: (For sentence tokenization)

Python

python -c "import nltk; nltk.download('punkt')"
🚀 Pipeline Quick Start
Ensure your input data (urls.xlsx or data.csv) is in the data/ directory. The entire pipeline runs through sequential script execution.

Run Data Preparation (Phases 1-3):

Execute the pipeline scripts sequentially from the project root:

Bash

# 1. Scraping & Parsing (Creates extracted_content.csv)
python streamlit_app/utils/parser.py data/urls.xlsx data/extracted_content.csv

# 2. Feature Engineering (Creates features.csv)
python streamlit_app/utils/features.py data/extracted_content.csv data/features.csv

# 3. Duplicate Detection (Creates duplicates.csv)
python streamlit_app/utils/duplicates.py data/features.csv data/duplicates.csv
Run Model Training & Analysis (Phase 4):

Bash

# 4. Model Training (Creates quality_model.pkl)
python streamlit_app/utils/scorer.py data/features.csv streamlit_app/models/quality_model.pkl

(Note: The full step-by-step analysis is contained in notebooks/seo_pipeline.ipynb )

🧠 Key Decisions

HTML Parsing: Used BeautifulSoup with the lxml parser for efficient content extraction, focusing on structural tags while removing HTML markup.




Feature Engineering: TF-IDF vectors were selected for document embeddings over Sentence Transformers, prioritizing computational simplicity and model stability on the small dataset.


Similarity Threshold Rationale: A conservative threshold of 0.80 was chosen for cosine similarity to identify genuine near-duplicate content.



Model Selection: Logistic Regression was chosen for the quality classifier due to its interpretability and effectiveness on engineered numerical features.
