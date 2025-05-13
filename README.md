# Fact-Checking System with LLM and BM25 (ON-GOING)

This project implements a basic fact-checking system that leverages the BM25 retrieval algorithm for finding relevant claims and a Large Language Model (LLM) for generating a concise fact-checked response. It's designed to take a user query and determine the veracity of a related claim based on a preprocessed dataset.

## Features

* **Data Loading & Preprocessing**: Loads the LIAR dataset and preprocesses text columns by converting to lowercase, removing punctuation, and stripping stopwords.
* **BM25 Retrieval**: Utilizes the BM25Okapi algorithm to efficiently search for the most relevant claims from the dataset based on a given query.
* **LLM Integration**: Uses a Groq API (specifically `meta-llama/llama-4-scout-17b-16e-instruct`) to synthesize a fact-checking response based on the retrieved claim's label, context, and speaker.
* **Command-Line Interface**: Provides a simple way to query the system via the command line.

## Setup and Installation

1.  **Clone the repository (if applicable) or ensure all files are in the same directory.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install pandas nltk rank_bm25 groq python-dotenv
    ```
4.  **Download NLTK data:**
    The script automatically attempts to download `stopwords` and `punkt` corpora when run. If you encounter issues, you can manually download them:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```
5.  **Obtain a Groq API Key:**
    * Sign up at [Groq](https://groq.com/).
    * Generate an API key.
6.  **Set up Environment Variable:**
    Create a `.env` file in the root directory of your project and add your Groq API key:
    ```
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    ```
7.  **Dataset:**
    Ensure you have the `train.tsv` file (from the LIAR dataset) in the same directory as your script. The script expects the LIAR dataset structure.

## Usage

1.  **Run the Jupyter Notebook:**
    Execute all cells in `I2SC_task.ipynb` sequentially. This will:
    * Load and preprocess the data.
    * Save the preprocessed data to `preprocessed_train.csv`.
    * Initialize the BM25 index.
    * Demonstrate an example query.

2.  **Run from the command line (after running the notebook cells at least once to generate `preprocessed_train.csv`):**
    ```bash
    python -c "import sys, os, pandas as pd, nltk; from nltk.tokenize import word_tokenize; from rank_bm25 import BM25Okapi; from groq import Groq; from dotenv import load_dotenv; load_dotenv(); train_df = pd.read_csv('preprocessed_train.csv'); train_df['statement'] = train_df['statement'].astype(str); tokenized_statements = [word_tokenize(statement) for statement in train_df['statement']]; bm25 = BM25Okapi(tokenized_statements); api_key = os.environ.get('GROQ_API_KEY'); client = Groq(api_key=api_key); def retrieve_claim(query, top_k=3): query_tokens = word_tokenize(query.lower()); scores = bm25.get_scores(query_tokens); top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]; results = []; for idx in top_indices: results.append({'statement': train_df.iloc[idx]['statement'], 'label': train_df.iloc[idx]['label'], 'context': train_df.iloc[idx]['context'], 'speaker': train_df.iloc[idx]['speaker'], 'score': scores[idx]}); return results; def generate_fact_checking_response(query): retrieved_claims = retrieve_claim(query, top_k=1); if not retrieved_claims: return \"I'm sorry, I couldn't find relevant information to fact-check this statement.\"; claim = retrieved_claims[0][\"statement\"]; label = retrieved_claims[0][\"label\"]; context = retrieved_claims[0][\"context\"]; speaker = retrieved_claims[0][\"speaker\"]; system_prompt = \"\"\"You are a fact-checking assistant. Your task is to determine if a provided claim is True or False, based strictly on the provided claim, label, context, and speaker. Provide a direct, brief response in the format: \"If you are referring to a claim by [speaker], it is categorically [True/False].\" Do not use external knowledge. Focus only on the provided information.\"\"\"; user_prompt = f\"\"\"Claim: \"{claim}\"\nLabel: \"{label}\"\nContext: \"{context}\"\nSpeaker: \"{speaker}\"\nQuery: \"{query}\"\n\nTask: Based strictly on the provided information (Claim, Label, Context, Speaker), determine whether the claim is \"True\" or \"False.\" Provide a direct response in the following format:\n\n\"If you are referring to a claim by [speaker], it is categorically [True/False].\"\n\nEnsure the response is clear, brief, and focuses only on the provided information.\"\"\"; messages = [{\"role\": \"system\", \"content\": system_prompt}, {\"role\": \"user\", \"content\": user_prompt}]; try: completion = client.chat.completions.create(model=\"meta-llama/llama-4-scout-17b-16e-instruct\", messages=messages, max_tokens=200, temperature=0.1); return completion.choices[0].message.content.strip(); except Exception as e: print(f\"An error occurred: {e}\"); return \"An error occurred while processing the request.\"; response = generate_fact_checking_response(sys.argv[1]); print(\"System response:\", response)" "Is the claim that the graduation rate in Wisconsin has improved since Republicans took over true?"
    ```
    (Note: The above command is a simplified one-liner for demonstration if the `.ipynb` cells were run. For a more robust CLI, you would typically save the `main()` function to a separate Python file, e.g., `fact_checker.py`, and run `python fact_checker.py "Your query here"`.)

    **Example:**
    ```bash
    python fact_checker.py "Is the claim that the graduation rate in Wisconsin has improved since Republicans took over true?"
    ```
    (Assuming you create `fact_checker.py` with the `main` and supporting functions.)

## Technologies Used

* **Python 3.x**
* **Pandas**: For data manipulation.
* **NLTK**: For text preprocessing (tokenization, stopwords).
* **rank_bm25**: For efficient claim retrieval.
* **Groq API**: For integrating with Large Language Models.

## Dataset

This project uses the [LIAR: A Benchmark Dataset for Fake News Detection](https://www.cs.ucsb.edu/~william/papers/liar_2017.pdf) dataset, specifically the `train.tsv` file.
