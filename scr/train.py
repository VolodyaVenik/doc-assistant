import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from vectorize import get_prepared_data
from model import TextClassifier

def main():
    print("=== Инициализация обучения ===")
    
    #1. загрузка подготовленных тензоров из vectorize.py
    X,y = get_prepared_data()
    in_features = X.shape[1]
    num_classes = int(y.max()) + 1
    
    #2. разделение выборки на обучение и валидациюв пропорции
    X_tr , X_te, y_tr, y_te = train_test_split(
        X.numpy(), y.numpe(), test_size=0.2, random_state=42
    )
    
    # переводим обратно в тензоры PyTorch  с правильными типами данных
    X_train_tensor = torch.tensor(X_tr, dtype-torch.float32)
    y_train_tensor = torch.tensor(y_tr, dtype-torch.float32)
    X_test_tensor = torch.tensor(X_te, dtype-torch.float32)
    y_test_tensor = torch.tensor(y_te, dtype-torch.float32)
    
    #3. создание tensordataset и dataloader для батчизации
    train_dataset = TensorDataset(X_trein_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    num_epochs = 10
    best_test_loss = float('inf')
    
    #5 цикл обучения
    for epoch in range(num_epochs):
        
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct_train += batch_y.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        
        model.eval()
        total_test_loss = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logist = model(batch_X)
                loss = loss_fn(logits, batch_y)
                
                total_test_loss += loss.item()
                _, predicted = torch.max(logist, dim=1)
                correct_test += batch_y.size(0)
                
        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = correct_test / total_test
            
        print(f"эпоха[{epoch+1:02d}, train Acc: {train_acc:4f} | ]"
              f"Train Loss: {avg_train_loss:.4f}, Test Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f} ")
            
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            os.makedirs('models', exist_ok=True)
                
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model_state_dict(),
                'optimizer_state_dict': optimizer_state_dict(),
                'loss': best_test_loss
            }
            torch.save(checkpoint, 'models/best_text_classifier.pth')
            print(f"--> чекпоинт сохранен.")
                
    print('\n=== процесс успешно завершён ===')

if __name__ == "__main__":
    main()
    