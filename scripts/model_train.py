import pandas as pd
from sklearn.naive_bayes import GaussianNB
import torch
import torch.nn as nn
import torch.optim as optim

def load_data() :
     file_path = "data/cleaned_merged_heart_dataset.csv"
     try:
        data = pd.read_csv(file_path)
        return data
     except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None

def train_model(x_normalized, y) :
    model = GaussianNB()
    model.fit(x_normalized, y)
    return model
 
class torchModel(nn.Module) :
   def __init__(self, input_size): 
       super(torchModel, self).__init__()
       self.fc1 = nn.Linear(input_size, 64)
       self.relu = nn.ReLU()
       self.fc2 = nn.Linear(64,1)
       self.sigmoid = nn.Sigmoid()
   
   def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      return x
   
def train_torch_model(x, y, input_size, epochs=10, learning_rate=0.01):
    model = torchModel(input_size)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x_train_tensor = torch.tensor(x, dtype=torch.float32)
    y_train_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  
        outputs = model(x_train_tensor)  
        loss = criterion(outputs, y_train_tensor)  
        loss.backward()  
        optimizer.step()  
    return model



