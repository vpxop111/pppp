from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import pickle
from pydantic import BaseModel
import os

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the RNN model
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

# Load vectorizer and model
try:
    with open("vectt.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    model = SMSRNN(input_size=5000, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load("scams.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Error loading files: {e}")
    raise HTTPException(status_code=500, detail="Error loading model or data files")

class Message(BaseModel):
    message: str

@app.post("/predict")
async def predict(message: Message):
    try:
        transformed_message = vectorizer.transform([message.message]).toarray()
        tensor_data = torch.tensor(transformed_message, dtype=torch.float32).unsqueeze(1)
        output = model(tensor_data)
        prediction = (output.squeeze().item() > 0.5)
        result = "scam" if prediction else "not scam"
        return {"prediction": result}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
