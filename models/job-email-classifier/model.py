import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the email dataset
print("\nLoading email dataset...")
df = pd.read_csv('email_data.csv')  # Replace with your actual CSV filename

print(f"\nDataset loaded successfully!")
print(f"Total emails: {len(df)}")

# Display basic information
print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Handle missing values
df['content'] = df['content'].fillna('')

# Analyze label distribution
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)
label_counts = df['job_type'].value_counts()
print(f"\n{label_counts}")

# Calculate ratio
total = len(df)
for label, count in label_counts.items():
    percentage = (count / total) * 100
    print(f"{label}: {count} emails ({percentage:.2f}%)")

# Calculate imbalance ratio
job_count = label_counts.get('job', 0)
not_job_count = label_counts.get('not_job', 0)
imbalance_ratio = not_job_count / job_count if job_count > 0 else 0
print(f"\nImbalance Ratio (not_job:job) = {imbalance_ratio:.2f}:1")
print(f"This is an IMBALANCED dataset - minority class (job) is only {(job_count/total)*100:.1f}%")

# Prepare labels
df['label'] = df['job_type'].map({'job': 1, 'not_job': 0})

# Split data (80% train, 20% test)
print("\n" + "=" * 60)
print("SPLITTING DATA")
print("=" * 60)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['content'].values,
    df['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Training set: {len(train_texts)} emails")
print(f"Test set: {len(test_texts)} emails")

# Initialize BERT tokenizer
print("\n" + "=" * 60)
print("INITIALIZING BERT TOKENIZER")
print("=" * 60)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("BERT tokenizer loaded!")

# Create custom dataset class
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders
print("\n" + "=" * 60)
print("CREATING DATASETS AND DATALOADERS")
print("=" * 60)

BATCH_SIZE = 16
MAX_LENGTH = 128

train_dataset = EmailDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
test_dataset = EmailDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# Load BERT model
print("\n" + "=" * 60)
print("LOADING BERT MODEL")
print("=" * 60)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)
print("BERT model loaded and moved to device!")

# Calculate class weights for imbalanced data
class_counts = np.bincount(train_labels)
class_weights = len(train_labels) / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"\nClass weights for imbalanced data:")
print(f"  not_job (0): {class_weights[0]:.2f}")
print(f"  job (1): {class_weights[1]:.2f}")

# Setup optimizer and scheduler
EPOCHS = 3
LEARNING_RATE = 2e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Loss function with class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training function
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            
            _, preds = torch.max(logits, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return (correct_predictions.double() / len(data_loader.dataset), 
            np.mean(losses), 
            predictions, 
            true_labels)

# Training loop
print("\n" + "=" * 60)
print("TRAINING BERT MODEL")
print("=" * 60)

best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 60)
    
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')
    
    val_acc, val_loss, _, _ = eval_model(model, test_loader, device)
    print(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}')
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_bert_email_classifier.pth')
        print(f'Best model saved with accuracy: {best_accuracy:.4f}')

# Final evaluation
print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

model.load_state_dict(torch.load('best_bert_email_classifier.pth'))
_, _, y_pred, y_test = eval_model(model, test_loader, device)

print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.2%}")

print("\nDetailed Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['not_job', 'job'],
    digits=4
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives (not_job correctly classified): {cm[0][0]}")
print(f"False Positives (not_job incorrectly classified as job): {cm[0][1]}")
print(f"False Negatives (job incorrectly classified as not_job): {cm[1][0]}")
print(f"True Positives (job correctly classified): {cm[1][1]}")

# Prediction function
def predict_email(text):
    """Predict if an email is job-related or not using BERT"""
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100
    
    result = "job" if prediction == 1 else "not_job"
    return result, confidence

# Test predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample_indices = np.random.choice(len(test_texts), min(5, len(test_texts)), replace=False)
for i, idx in enumerate(sample_indices, 1):
    email = test_texts[idx]
    actual = 'job' if test_labels[idx] == 1 else 'not_job'
    prediction, confidence = predict_email(email)
    
    print(f"\nSample {i}:")
    print(f"Content: {email[:100]}...")
    print(f"Actual: {actual}")
    print(f"Predicted: {prediction} (Confidence: {confidence:.2f}%)")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nModel saved as 'best_bert_email_classifier.pth'")
print("To use the model later, load it with:")
print("model.load_state_dict(torch.load('best_bert_email_classifier.pth'))")