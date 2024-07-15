from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import logging

# Suppress InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Configure logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)

app = FastAPI()

# Load the model and vectorizer
model_pat = 'scams.pth'
vectorizer_path = 'vectt.pkl'

# Define the RNN model class
class SMSRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SMSRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        return out

def load_model_and_vectorizer(model_pat, vectorizer_path):
    # Load the model
    model = SMSRNN(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(model_pat))
    model.eval()

    # Load the vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# Global variables
input_size = 5000  # Assuming TF-IDF vectorizer with max_features=5000
hidden_size = 128
num_layers = 2

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer(model_pat, vectorizer_path)

# Prediction functions
def preprocess_message(message, vectorizer):
    system_message = f"System: This message needs to be classified as scam or ham. Message: {message}"
    message_vectorized = vectorizer.transform([system_message])
    message_tensor = torch.tensor(message_vectorized.toarray(), dtype=torch.float32).unsqueeze(1)
    return message_tensor

def predict_message(model, message_tensor):
    model.eval()
    with torch.no_grad():
        output = model(message_tensor)
        prediction = (output > 0.5).float().item()
    return prediction

class Message(BaseModel):
    message: str

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains in production
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post('/predict')
async def predict(message: Message = Body(...)):
    try:
        # Preprocess the message
        message_tensor = preprocess_message(message.message, vectorizer)

        # Predict using the model
        prediction = predict_message(model, message_tensor)

        # Interpret prediction
        result = 'scam' if prediction == 1 else 'ham'

        # Return JSON response
      
        return {'message': message.message, 'predicted_result': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
