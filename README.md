# Medical ChatBOT

This project is a Medical ChatBOT designed to provide accurate and reliable medical information using the Mistral-7B-Instruct-v0.3 model. The chatbot uses embeddings from the `sentence-transformers/all-MiniLM-L6-v2` model and stores the data in a FAISS vector store for fast and efficient retrieval.

## Model Details

- **Model Name**: Mistral-7B-Instruct-v0.3
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Model Size**: 7 Billion Parameters

## How It Works

1. **Data Loading**: The chatbot loads medical data from PDF files and stores the embeddings in a FAISS vector store.
2. **Query Processing**: When a user asks a question, the chatbot retrieves relevant documents from the vector store.
3. **Response Generation**: The chatbot uses the Mistral-7B-Instruct-v0.3 model to generate a response based on the retrieved documents and the user's query.
4. **Fast and Accurate**: The use of vector stores ensures fast retrieval of relevant documents, and the powerful Mistral-7B-Instruct-v0.3 model provides accurate and reliable responses.

## How to Use

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/Roksana18cse04/Medical_Chat_BOOT.git
    cd medical-chatbot
    ```

2. **Create and activate a virtual environment**:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:

    Create a `.env` file in the root directory and add your Hugging Face token:

    ```env
    HF_TOKEN=your_huggingface_token
    ```

### Running the ChatBOT

1. **Start the Streamlit app**:

    ```sh
    streamlit run Medical_ChatBOOT.py
    ```

2. **Interact with the ChatBOT**:

    Open your web browser and go to `http://localhost:8501`. You can now interact with the Medical ChatBOT by typing your queries in the input box.

## Example Usage

1. **User Query**: "What is the treatment for a broken bone?"
2. **ChatBOT Response**: The chatbot retrieves relevant documents from the vector store and uses the Mistral-7B-Instruct-v0.3 model to generate a detailed response, including definitions, summaries, and step-by-step guides.

## Project Structure

- `Medical_ChatBOOT.py`: Main script to run the Streamlit app.
- `requirements.txt`: List of dependencies.
- `.env`: Environment variables file (not included in version control).
- `vectorstore/`: Directory containing the FAISS vector store.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
