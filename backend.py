import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr
import pyttsx3

# Initialize recognizer and TTS engine
engine = pyttsx3.init()

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def speech_to_text():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    with sr.Microphone(sample_rate=16000) as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=6)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio, language="en-in")
        print("You said:", text)
        engine.say(f"You said: {text}")
        engine.runAndWait()
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
        engine.say("Sorry, I couldn't understand the audio.")
        engine.runAndWait()
        return ""
    except sr.RequestError as e:
        print(f"API error: {e}")
        engine.say(f"API error: {e}")
        engine.runAndWait()
        return ""


# Load dataset
with open(r'C:\Users\Neeraj Kumar\1st Hackathon\dataset2 (1).json') as f:
    intents = json.load(f)["intents"]

# Tokenization (Simplified)
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.idx = 2  # Reserving 0 and 1 for padding and unknown tokens

    def fit(self, texts):
        for text in texts:
            for word in text.split():
                if word.lower() not in self.vocab:
                    self.vocab[word.lower()] = self.idx
                    self.idx += 1

    def encode(self, text, max_length=20):
        tokens = [self.vocab.get(word.lower(), 1) for word in text.split()]  # 1 for unknown tokens
        return tokens[:max_length] + [0] * (max_length - len(tokens))


tokenizer = SimpleTokenizer()
patterns = [p for intent in intents for p in intent['patterns']]
tokenizer.fit(patterns)


# Dataset class
class MentalHealthDataset(Dataset):
    def __init__(self, intents, tokenizer):
        self.data = [(tokenizer.encode(p), intent["tag"]) for intent in intents for p in intent['patterns']]
        self.labels = LabelEncoder()
        self.labels.fit([intent["tag"] for intent in intents])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        # Ensure labels are returned as Long type
        return torch.tensor(tokens), torch.tensor(self.labels.transform([label])[0], dtype=torch.long)


# Model
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])


# Hyperparameters and Training Setup
vocab_size = tokenizer.idx
embed_dim = 64
hidden_dim = 128
output_dim = len(set([intent["tag"] for intent in intents]))

model = SimpleLSTM(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader
dataset = MentalHealthDataset(intents, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training Loop
def train_model():
    model.train()
    for epoch in range(200):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            # Ensure target labels are of type Long (int64)
            loss = criterion(output, y.long())  # Convert y to Long type here
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

train_model()


# Inference (Chat)
def chat():
    model.eval()
    while True:
        print("You: ")
        user_input = speech_to_text()
        if user_input.lower() == "exit":
            exit_greet = "Feel free to reach out anytime, later. Thank you!!"
            text_to_speech(exit_greet)
            break
        encoded = torch.tensor(tokenizer.encode(user_input)).unsqueeze(0)
        output = model(encoded)
        predicted_label = torch.argmax(output, dim=1)
        tag = dataset.labels.inverse_transform(predicted_label.cpu().numpy())[0]
        for intent in intents:
            if intent["tag"] == tag:
                response = random.choice(intent['responses'])
                print(f"Mental Health Bot: {response}")
                text_to_speech(response)

chat()
