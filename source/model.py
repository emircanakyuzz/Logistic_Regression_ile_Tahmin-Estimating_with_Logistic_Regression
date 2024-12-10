from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Veri İşleme ve Veri Seti
from data_preprocessing import X, y

class LogisticRegressionModel:
    def __init__(self, X, y, test_size=0.25, random_state=3):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.model = LogisticRegression()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.y_pred = None

    def split_data(self):
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred_prob = self.model.predict_proba(self.X_test)
        self.y_pred = np.argmax(y_pred_prob, axis=1)
        return self.y_pred

    def evaluate(self):
        if self.y_pred is None:
            raise ValueError("Model önce tahmin yapmalıdır. predict() metodu çağrılmalı.")
        
        accuracy = accuracy_score(self.y_pred, self.y_test)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, self.y_pred)
        return accuracy, f1, cm

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.title('Confusion Matrix')
        plt.show()

# Logistic Regression Modelini Çalıştırma
if __name__ == "__main__":
    model = LogisticRegressionModel(X, y)
    model.train()
    model.predict()
    accuracy, f1, cm = model.evaluate()

    print(f"Test Veri Setindeki Doğruluk Oranımız (Test Accuracy): {accuracy}")
    print("Confusion Matrix:")
    print(cm)
    print(f"F1-Score (Weighted): {f1}")

    model.plot_confusion_matrix(cm)