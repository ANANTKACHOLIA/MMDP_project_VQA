import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import QADataset
from model import VQAModel
from utils import load_statements, split_df

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
batch_size = 32
learning_rate = 1e-5
num_epochs = 200

# Load and split the dataset
statements_file = 'statements.txt'  # File containing statements
train_csv = 'train.csv'  # Output train CSV file
test_csv = 'test.csv'    # Output test CSV file
df = load_statements(statements_file)
split_df(df, train_out=train_csv, test_out=test_csv)

# Load datasets
train_dataset = QADataset(csv_file=train_csv, ...)
test_dataset = QADataset(csv_file=test_csv, ...)

# Create DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = VQAModel(...)
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop (continued)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, questions, answers in train_loader:
        images, questions, answers = images.to(device), questions.to(device), answers.to(device)
        
        # Forward pass
        outputs = model(images, questions)
        loss = criterion(outputs, answers)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, questions, answers in test_loader:
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)
            
            # Forward pass
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += answers.size(0)
            total_correct += (predicted == answers).sum().item()
    
    # Print validation loss and accuracy
    print(f"Validation Loss: {val_loss/len(test_loader):.4f}, Accuracy: {(total_correct / total_samples) * 100:.2f}%")
