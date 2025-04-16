PDF-based Question Answering System
A lightweight NLP system that processes PDF documents to answer user questions using cosine similarity and a custom-trained neural network.

Features
PDF Text Extraction: Parallel processing of multiple PDF files

Text Preprocessing: Tokenization, stopword removal, and stemming

Neural Model: Simple bidirectional LSTM architecture (SimpleLLM)

Similarity-based Responses: Cosine similarity matching between questions and document content

Multiprocessing: Efficient PDF processing using 4 parallel workers

Installation
bash
pip install torch nltk pypdf2 numpy
Usage
Prepare PDFs:

Place all documents in a single folder

Ensure files have .pdf extension

Run the system:

bash
python main.py
Enter the path of your folder containing PDFs: /path/to/your/pdf/folder
Ask questions:

bash
You: What are the key findings in this research?
Bot: [System returns relevant sentences from documents]
Exit:

bash
Type "exit" or "quit" to end session
Project Structure
