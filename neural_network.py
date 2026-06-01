import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np

# Log through uvicorn's logger so model messages match the server's clean format.
logger = logging.getLogger("uvicorn")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset



class SimpleClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)



class ModelService:

    def __init__(self):

        self.model_path = "model.pth"
        self.scaler_path = "scaler.pkl"

        self.model = SimpleClassifier()
        self.scaler = StandardScaler()

        self.classes = load_iris().target_names.tolist()

        self.load_or_train()


    def train_model(self):

        print("Training new model...")

        iris = load_iris()
        X = iris.data
        y = iris.target

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # SCALE
        X_train = self.scaler.fit_transform(X_train)

        # TENSORS
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        # DATA LOADER (IMPORTANT UPGRADE)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # TRAIN LOOP (BETTER + FASTER CONVERGENCE)
        for epoch in range(200):

            total_loss = 0

            for batch_x, batch_y in loader:

                optimizer.zero_grad()

                outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                logger.info("Epoch %d: loss %.4f", epoch, total_loss / len(loader))

        self.save_model()



    def save_model(self):
        # save as pk1
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        logger.info("Iris model trained and saved.")



    def load_model(self):
        # Saved as pk1. weights_only=True loads just the tensor state_dict and
        # silences torch's pickle FutureWarning (we control the file).
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.eval()

        self.scaler = joblib.load(self.scaler_path)

        logger.info("Iris model loaded.")


    def load_or_train(self):

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):

            self.load_model()

        else:

            self.train_model()
            self.load_model()



    def predict(self, features):

        x = np.array(features).reshape(1, -1)
        x = self.scaler.transform(x)

        x_tensor = torch.tensor(x, dtype=torch.float32)

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(x_tensor)

            probs = torch.softmax(outputs, dim=1)

            conf, pred = torch.max(probs, 1)

        return {
            "prediction": self.classes[pred.item()],
            "confidence": round(conf.item(), 4),
            "probabilities": {
                self.classes[i]: round(probs[0][i].item(), 4)
                for i in range(3)
            }
        }



