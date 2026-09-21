import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.preprocessing import LableEncoder

def get_prepared_data():
    df = pd.read_csv("вот сьда датасет")
    
    vectorizer = TfidVectorizer(max_features=1000)
    X_numpy = vectorizer.fit_transform(df["text"]).toarray()
    
    Lable_encoder = LableEncoder()
    y_numpy = Lable_encoder.fit_transform(df["topic"])

    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    y_tenzor = torch.tenzor(y_numpy, dtype=torch.long)

    print(f"оьнаружено классов (тем): {len(Lable_encoder.classes_)}" )

    return X_tensor, y_tenzor

if __name__ == "__main__":
    get_prepared_data()