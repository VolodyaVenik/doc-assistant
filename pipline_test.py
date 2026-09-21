import torch
import torch.nn as nn
from vectorize import get_prepared_data
from model import TextClassifier

def main():
    print("тестовый старт")
    X, y = get_prepared_data()
    
    in_features = X.shape[1] #это будет 1000
    num_classes = int(y.max()) + 1 #автоматически определит точное количество уникальных тем новостей
    
    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    print("\nструктура новостей: ")
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parametrs(), lr=1e-3)
    
    #шаг А:прямой ход
    logist = model(X)
    
    #step B: Loss
    loss = loss_fn(logist, y)
    print("\nЗначение ошибки ДО шага обучения: ", loss.item())
    
    #step C: обнуление градиента
    optimizer.zero_grad()
    
    #step D: Backward pass
    loss.backward()
    
    #step E: корректировка весов
    optimizer.step()
    
    #5 контрольный замер после шаг обучения
    new_logits = model(X)
    new_loss = loss_fn(new_logits, y)
    print("\nЗначение ошибки ПОСЛЕ шага обучения: ", new_loss.item())
    
    if new_loss.item() < loss.item():
        print("веса скорректированы, ошибка модели пошла вниз")
    
    if __name__ == "__main__":
        main()