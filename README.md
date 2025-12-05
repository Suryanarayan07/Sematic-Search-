Here is a comprehensive `README.md` file tailored for your GitHub repository. It includes project description, installation instructions, usage, and an explanation of the methodology used in your code.

-----

# Semantic Search Analysis: SigLIP vs. MPNet

## 📋 Project Overview

This project performs a comparative analysis of semantic search models on the **NFCorpus** (Nutrition Facts Corpus) medical dataset. It evaluates the retrieval performance of **SigLIP** (a Vision-Language model adapted for text) against **MPNet** (a state-of-the-art text embedding model).

The system handles the full pipeline:

1.  **Data Loading:** Automatically downloads and parses the BEIR/NFCorpus dataset.
2.  **Embedding:** Generates dense vector representations using Transformer models.
3.  **Indexing:** Uses **FAISS** (Facebook AI Similarity Search) for efficient vector retrieval.
4.  **Evaluation:** Computes **MAP@25** (Mean Average Precision) to quantify performance.
5.  **Error Analysis:** Generates detailed reports on False Positives (FP) and False Negatives (FN) to understand model behavior.

## ✨ Key Features

  * **Multi-Model Support:** Compare specialized text models (MPNet, MiniLM) against multimodal encoders (SigLIP).
  * **Robust Evaluation:** Implements standard Information Retrieval metrics (Precision, Recall, MAP@K).
  * **Deep Error Analysis:** Automated "post-mortem" of search results to identify why relevant documents were missed or why irrelevant ones were retrieved.
  * **Efficient Search:** Utilizes `faiss.IndexFlatIP` for fast inner-product similarity search.
  * **Automatic Handling:** Handles dataset downloading, unzipping, and preprocessing automatically.

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/semantic-search-analysis.git
    cd semantic-search-analysis
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install torch numpy pandas tqdm faiss-cpu transformers sentence-transformers beir
    ```

    *(Note: If you have a GPU, ensure you install the CUDA version of PyTorch).*

## 🚀 Usage

To run the complete analysis pipeline, simply execute the main script:

```bash
python main.py
```

### What happens when you run this?

1.  The script checks for the **NFCorpus** dataset in `./datasets`. If missing, it downloads it.
2.  It loads the **SigLIP** model, encodes the corpus, builds a FAISS index, and runs queries.
3.  It repeats the process for **MPNet** (or MiniLM as fallback).
4.  It calculates MAP@25 scores for both models.
5.  It performs a comparative FP/FN analysis.
6.  Results are saved to the `results/` directory.

## 📂 Project Structure

```
├── main.py                 # Primary entry point; orchestrates the full pipeline
├── data_loader.py          # Handles BEIR dataset downloading and loading
├── embedding_models.py     # Wrapper classes for Transformer models
├── indexing_retrieval.py   # FAISS index implementation
├── evaluation.py           # MAP@K and precision/recall logic
├── fp_fn_analysis.py       # Logic for detailed False Positive/Negative auditing
├── datasets/               # Directory where NFCorpus is downloaded
└── results/                # Output directory for analysis JSONs
```

## 🧠 Methodology

### The Models

  * **SigLIP (google/siglip-base-patch16-256-multilingual):** A CLIP-style model. We test its capabilities as a pure text retriever. *Note: Includes custom handling for SigLIP's max token length (64).*
  * **MPNet (sentence-transformers/all-mpnet-base-v2):** A general-purpose text embedding model trained on large text pairs, serving as the strong baseline.

### The Metric: MAP@25

We use **Mean Average Precision at K=25**. This metric considers the rank of relevant documents. A relevant document at rank 1 contributes more to the score than a relevant document at rank 10.

### Error Analysis

The system identifies:

  * **False Positives:** Documents ranked highly but marked irrelevant in ground truth (often due to lexical overlap without semantic meaning).
  * **False Negatives:** Relevant documents missed by the model (often due to vocabulary mismatch, e.g., "cancer" vs "neoplasm").

## 📊 Sample Output

After execution, check `results/summary.json` for a structured overview:

```json
{
  "evaluation": {
    "SigLIP": {
      "MAP@25": 0.1234,
      "Mean_Precision": 0.2050
    },
    "MPNet": {
      "MAP@25": 0.3456,
      "Mean_Precision": 0.4500
    }
  }
}
```

## 🤝 Contributing

Contributions are welcome\! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

-----

*This project uses the [BEIR Benchmark](https://github.com/beir-cellar/beir) for dataset handling.*