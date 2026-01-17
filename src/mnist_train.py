import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# 1. Load Data
# Kaggle MNIST train.csv has 'label' as 1st col, pixels 0-783 as rest
print("Loading dataset...")

# Define paths assuming script will reside in src/ but run from project root
# Or if run from src, we need ../data. 
# Best practice: use absolute paths relative to script location or run from root.
# We will assume running from project root: python src/mnist_train.py

TRAIN_PATH = 'data/train.csv'
if not os.path.exists(TRAIN_PATH):
    # Try ../data if running from src folder
    TRAIN_PATH = '../data/train.csv'

if not os.path.exists(TRAIN_PATH):
    print(f"Error: {TRAIN_PATH} not found!")
    exit(1)

# Read only a subset if needed for speed, but full fine for RF
df = pd.read_csv(TRAIN_PATH)
print(f"Data shape: {df.shape}")

y = df['label']
X = df.drop('label', axis=1)

# Normalize pixel values
X = X / 255.0

# 2. EDA
print("\nPerforming EDA...")
# Ensure images directory exists
os.makedirs('images', exist_ok=True)

# Class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.title("Digit Class Distribution")
plt.savefig('images/class_distribution.png')
print("Saved images/class_distribution.png")

# Visualize some samples
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # Reshape 1D array (784 features) to 2D (28x28)
    img = X.iloc[i].values.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('images/sample_digits.png')
print("Saved images/sample_digits.png")

# 3. Model Training
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier (this may take a minute)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. Evaluation
print("\nEvaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('images/confusion_matrix.png')
print("Saved images/confusion_matrix.png")

# 5. Save Model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/mnist_model.pkl')
print("\nModel saved to 'models/mnist_model.pkl'")

# 6. Generate Submission
print("\nGenerating submission.csv...")
TEST_PATH = 'data/test.csv'
if not os.path.exists(TEST_PATH):
    TEST_PATH = '../data/test.csv'

if os.path.exists(TEST_PATH):
    test_df = pd.read_csv(TEST_PATH)
    # Normalize
    X_submission = test_df / 255.0
    
    # Predict
    submission_preds = model.predict(X_submission)
    
    # Create DataFrame
    submission = pd.DataFrame({
        'ImageId': range(1, len(submission_preds) + 1),
        'Label': submission_preds
    })
    
    # Save
    submission.to_csv('submission.csv', index=False)
    print(f"Submission file saved as 'submission.csv' with {len(submission)} predictions.")
else:
    print(f"Warning: {TEST_PATH} not found. Skipping submission generation.")
