import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import PyPDF2
from multiprocessing import Pool
from collections import Counter
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the SimpleLLM model
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=2):  # Fixed method name
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Extract text from a single PDF file
def extract_text_from_file(file_path):
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() if page.extract_text() else ""
            if len(page_text.split()) > 10:  # Skip pages with fewer than 10 words
                text += page_text.lower()
    return text

# Extract text from all PDFs in a folder (parallelized)
def extract_text_from_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    with Pool(processes=4) as pool:  # Use 4 parallel processes
        texts = pool.map(extract_text_from_file, files)
    return " ".join(texts)

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Tokenize text and prepare data for training
def tokenize_text(text, sequence_length=10):
    tokens = preprocess_text(text)
    word_counts = Counter(tokens)
    vocab = [word for word, count in word_counts.items() if count > 2]  # Keep words with count > 2
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    input_data = []
    for i in range(0, len(tokens) - sequence_length):
        input_data.append([word_to_idx[word] for word in tokens[i:i + sequence_length] if word in word_to_idx])
    return input_data, word_to_idx, idx_to_word

# Train model on text
def train_model_on_text(text, sequence_length=10, batch_size=32, num_epochs=10):
    input_data, word_to_idx, idx_to_word = tokenize_text(text, sequence_length)
    vocab_size = len(word_to_idx)
    embed_size = 32  # Reduced embedding size
    hidden_size = 64  # Reduced hidden size
    output_size = vocab_size

    model = SimpleLLM(vocab_size, embed_size, hidden_size, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    inputs = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in input_data], batch_first=True).to(device)
    targets = torch.tensor([seq[-1] for seq in input_data], dtype=torch.long).to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(inputs), batch_size):
            input_batch = inputs[i:i + batch_size]
            target_batch = targets[i:i + batch_size]
            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(inputs):.4f}')
    return model, word_to_idx, idx_to_word

# Compute cosine similarity
def compute_cosine_similarity(vector1, vector2):
    all_keys = set(vector1.keys()).union(set(vector2.keys()))
    aligned_vector1 = np.array([vector1.get(key, 0) for key in all_keys])
    aligned_vector2 = np.array([vector2.get(key, 0) for key in all_keys])
    dot_product = np.dot(aligned_vector1, aligned_vector2)
    norm1 = np.linalg.norm(aligned_vector1)
    norm2 = np.linalg.norm(aligned_vector2)
    return dot_product / (norm1 * norm2 + 1e-8)  # Add epsilon to avoid division by zero

# Generate response using cosine similarity
def generate_response_with_similarity(input_text, original_text, max_sentences=3):
    input_tokens = preprocess_text(input_text)
    input_vector = Counter(input_tokens)

    sentences = original_text.split('.')
    sentence_vectors = [
        (Counter(preprocess_text(sentence)), sentence.strip())
        for sentence in sentences if sentence.strip()
    ]

    scored_sentences = [
        (compute_cosine_similarity(input_vector, sentence_vector), sentence)
        for sentence_vector, sentence in sentence_vectors
    ]

    ranked_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
    top_sentences = [sentence for _, sentence in ranked_sentences[:max_sentences]]
    if top_sentences:
        return ' '.join(top_sentences)
    else:
        return "I don't have specific information on that. Could you try rephrasing?"

# Main function
def main():
    folder_path = input("Enter the path of your folder containing PDFs: ")
    text = extract_text_from_folder(folder_path)
    print("Training the model on the extracted text. This may take a while...")

    model, word_to_idx, idx_to_word = train_model_on_text(text)

    print("Training complete. You can now ask questions!")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = generate_response_with_similarity(question.lower(), text, max_sentences=3)
        print("Bot:", response)

if __name__ == "__main__":
    main()
