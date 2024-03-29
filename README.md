# RAG Content Generation with GPT-2

This repository demonstrates a Retrieval-Augmented Generation (RAG) system using the GPT-2 language model for content generation. The RAG system combines information retrieval with language generation, allowing the generation of contextually relevant content based on a query.

## Installation

Ensure you have the required libraries installed:

```bash
!pip install transformers faiss-cpu
```

# Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Disciplined-22/RAG-Content-Generation-with-GPT2.git
    cd RAG-Content-Generation-with-GPT2
    ```

2. **Run the example script:**

    ```bash
    python main.py
    ```

    This script initializes a vector database with random vectors, loads a pre-trained GPT-2 model, and demonstrates RAG content generation for a sample query.

# Components

1. **GPT-2 Model**

    - The code utilizes the GPT-2 language model from the Transformers library for language generation.

2. **Vector Database**

    - A vector database is created with random vectors (1000 vectors, each with 768 dimensions). This database simulates the retrieval component of the RAG system.

3. **Retrieval and Generation Functions**

    - Functions are implemented for information retrieval from the vector database (`retrieve_from_database`) and RAG content generation (`generate_rag_content`).

    The `retrieve_from_database` function uses the Faiss library to efficiently search and retrieve top-k similar vectors from the vector database based on a query vector.

    The `generate_rag_content` function tokenizes a given query using the GPT-2 tokenizer, obtains hidden states from the GPT-2 model, and then computes the mean of these hidden states to create a query vector. It utilizes the `retrieve_from_database` function to retrieve relevant vectors and generates content using the GPT-2 model.

    This combination of retrieval and generation functions showcases the RAG paradigm, where retrieved information enhances the context for language generation, resulting in more accurate and contextually relevant content.

