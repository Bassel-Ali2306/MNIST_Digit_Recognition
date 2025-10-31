# ============================================
# ML Assignment 2 – Part A
# ============================================

# --- IMPORTS & SETUP ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ============================================================
# PART A1 – DATA PREPARATION
# ============================================================

# Define a transform to convert MNIST images to tensors
transform = transforms.Compose([
    transforms.ToTensor()   # Converts PIL image to tensor with shape [C,H,W] and normalizes to [0,1]
])

# Load MNIST dataset
train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# Combine train and test into a single dataset of 70,000 samples
X = torch.cat([train_data.data, test_data.data], dim=0)
y = torch.cat([train_data.targets, test_data.targets], dim=0)

# Normalize pixel values to [0,1] and flatten (28×28 → 784)
X = X.float() / 255.0
X_flat = X.view(-1, 28 * 28)

# Perform stratified split: 60% train, 20% validation, 20% test
# `stratify=y` ensures each split has the same class distribution
X_train, X_temp, y_train, y_temp = train_test_split(X_flat, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Display sample counts for verification
print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# ============================================================
# PART A2 – LOGISTIC REGRESSION (BINARY 0 vs 1)
# ============================================================

# --- Helper function to filter specific digits (here 0 and 1) ---
def filter_digits(X, y, digits=(0, 1)):
    mask = (y == digits[0]) | (y == digits[1])
    return X[mask], y[mask]

# Filter the dataset for binary classification (digits 0 and 1 only)
X_train_bin, y_train_bin = filter_digits(X_train, y_train)
X_val_bin, y_val_bin = filter_digits(X_val, y_val)
X_test_bin, y_test_bin = filter_digits(X_test, y_test)

# Convert labels to float tensors (required for BCELoss)
# Add an extra dimension to make shape (N,1)
y_train_bin = y_train_bin.float().unsqueeze(1)
y_val_bin = y_val_bin.float().unsqueeze(1)
y_test_bin = y_test_bin.float().unsqueeze(1)

# Create PyTorch DataLoaders for batching
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_bin, y_train_bin), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_bin, y_val_bin), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_bin, y_test_bin), batch_size=batch_size, shuffle=False)

# Define Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Single linear layer: 784 input features → 1 output (probability of class 1)
        self.linear = nn.Linear(784, 1)

    def forward(self, x):
        # Apply sigmoid to convert logits to probability [0,1]
        return torch.sigmoid(self.linear(x))

# Initialize model, loss function, and optimizer
model = LogisticRegressionModel()
criterion = nn.BCELoss()                  # Binary Cross-Entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Gradient Descent optimizer

# Training the logistic regression model
epochs = 10
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()           # Reset gradients
        outputs = model(images)         # Forward pass
        loss = criterion(outputs, labels)  # Compute BCE loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Gradient descent update

        total_loss += loss.item()
        preds = (outputs >= 0.5).float()  # Convert probabilities to binary labels
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # Validation phase
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    # Store epoch metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# --- Plot Loss and Accuracy Curves ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend(); plt.title("Binary Logistic Regression – Loss")

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Validation Acc')
plt.legend(); plt.title("Binary Logistic Regression – Accuracy")
plt.show()

# --- Evaluate on Test Set ---
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = (outputs >= 0.5).float()
        all_preds.extend(preds.squeeze().numpy())
        all_labels.extend(labels.squeeze().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\nFinal Test Accuracy (0 vs 1): {test_acc:.4f}")

# Confusion matrix visualization
cm = confusion_matrix(all_labels, all_preds)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix – Binary Logistic Regression (0 vs 1)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# ============================================================
# PART A3 – SOFTMAX REGRESSION (MULTI-CLASS 0–9)
# ============================================================

# Convert labels to long tensors (required for CrossEntropyLoss)
y_train_long = y_train.long()
y_val_long = y_val.long()
y_test_long = y_test.long()

# Create dataloaders for full multiclass dataset
train_loader = DataLoader(TensorDataset(X_train, y_train_long), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val_long), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test_long), batch_size=batch_size, shuffle=False)

# --- Define Softmax Regression model ---
class SoftmaxRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 784 input features → 10 output classes
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        # CrossEntropyLoss automatically applies Softmax,
        # so we just return raw logits here.
        return self.linear(x)

# Initialize model, loss, and optimizer
model = SoftmaxRegressionModel()
criterion = nn.CrossEntropyLoss()        # Multi-class cross-entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --- Training Loop ---
epochs = 10
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(epochs):
    # ===== Training =====
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)            # Forward pass
        loss = criterion(outputs, labels)  # CrossEntropyLoss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)      # Class with max logit = predicted class
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    # ===== Validation =====
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# --- Plot Loss and Accuracy ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend(); plt.title("Softmax Regression – Loss")

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.legend(); plt.title("Softmax Regression – Accuracy")
plt.show()

# --- Evaluate on Test Set ---
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix for all 10 digits
cm = confusion_matrix(all_labels, all_preds)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix – Softmax Regression (0–9)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# --- Per-class accuracy and report ---
print("\nClassification Report (Softmax Regression):")
print(classification_report(all_labels, all_preds, digits=4))
