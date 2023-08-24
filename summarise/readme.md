# PDF File Summarizer

The PDF File Summarizer is a Python script that utilizes various language processing techniques to generate summaries of the content within a PDF file. This script combines multiple technologies, including OpenAI's GPT-3.5 model, text splitting, and a summarization chain, to create concise summaries from the provided PDF.

## How It Works

The script follows a series of steps to accomplish its summarization task:

1. **Environment Configuration**: The script starts by loading environment variables from a `.env` file. These variables include the API key and base URL for the OpenAI API. These credentials are necessary for making API calls.

2. **FileSummarizer Class**: The core functionality of the script is encapsulated within the `FileSummarizer` class. When an instance of this class is created, it requires the path to a PDF file as input.

3. **Azure API Configuration**: The class sets up the configuration for the OpenAI API by specifying that it will use the Azure engine with a specific API version and key.

4. **File Loader**: The script loads the specified PDF file using the PyPDFLoader, which is a part of the `langchain` library. The loader extracts the content from each page of the PDF.

5. **Text Splitting**: The content from each page is split into smaller chunks using the RecursiveCharacterTextSplitter. This is done to ensure that the summarization process can handle large amounts of text efficiently. Chunks of text are created with a defined size and overlap.

6. **Map-Reduce Summarization Chain**: The script loads a summarization chain using the `load_summarize_chain` function from the `langchain` library. The chain type used here is "map_reduce". This summarization chain is a series of language model interactions designed to create meaningful summaries.

7. **Running the Chain**: The script runs the summarization chain on the first two chunks of text that were generated. The result of this operation is a summarized output.

## Getting Started

1. Install the required libraries by running: `pip install openai python-dotenv`.

2. Make sure you have an OpenAI API key and Azure API endpoint. Update the `.env` file with your API key and endpoint.

3. Provide the path to the PDF file you want to summarize by setting the `file_path` variable.

4. Run the script. It will generate summaries of the PDF content using the specified pipeline.

## Example Usage

```python
# Path to the PDF file
file_path = "./input/budget_speech.pdf"

# Create a FileSummarizer instance
summarizer = FileSummarizer(file_path)

# Call the map_reduce_summary method to generate summaries
summarizer.map_reduce_summary()
```

## Note

- This script is a demonstration of how to utilize various language processing components to generate summaries from a PDF file.
- The summarization quality may vary based on the content and complexity of the PDF.
- This script can be further customized and optimized to suit specific summarization needs.

---