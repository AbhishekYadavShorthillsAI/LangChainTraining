
# Document Similarity Search using Embeddings

This project demonstrates how to perform a document similarity search using embeddings. It leverages OpenAI's GPT-3.5 language model for chat, HuggingFace's transformer models for text embeddings, and FAISS for efficient similarity search.

## Overview

The project is structured using an object-oriented programming (OOP) approach, with classes representing different components of the workflow.

- `OpenAIConfig`: Configures the OpenAI API using provided credentials.
- `DocumentProcessor`: Loads and processes PDF documents, splitting them into smaller chunks.
- `EmbeddingProcessor`: Creates an embedding database using HuggingFace's transformer models and FAISS.
- `DocumentSearch`: Performs similarity search on the embedding database.

## Prerequisites

- Python 3.6 or higher
- OpenAI API key
- Environment variables configured in a `.env` file:

  ```plaintext
  OPENAI_API_KEY=<your-api-key>
  OPENAI_API_BASE=https://api.openai.com/v1
  ```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure your environment variables by creating a `.env` file in the root directory with the OpenAI API key and base URL.

2. Run the main script:

   ```bash
   python main.py
   ```

   This will load, process, embed, and search for document similarities based on the given query.

## Customization

- You can modify the PDF file path in the `DocumentProcessor` class to process different documents.
- Adjust the `model_name` parameter in the `EmbeddingProcessor` class to use different HuggingFace transformer models.
- Change the query in the `search_similarity` method of the `DocumentSearch` class to find similarities for different terms.

