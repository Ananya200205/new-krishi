import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
# Dataset directory path
dataset_path = r"C:\Users\Ananya V Shetty\Desktop\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

# # Read the CSV file
# data = pd.read_csv(csv_path)

# Root directory path for image dataset
base_path = r"C:\Users\Ananya V Shetty\OneDrive\Documents\KRISHISUVIDHA"

# Prepare lists to store image data and labels
image_data = []
labels = []

# Preprocess images from dataset
for label in os.listdir(dataset_path):
     label_path = os.path.join(dataset_path, label)
     if os.path.isdir(label_path):  # Check if it's a directory
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                image = image.resize((128, 128))  # Resize to a fixed size
                image_array = np.array(image).flatten()  # Convert to 1D array
                
                image_data.append(image_array)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# Convert lists to NumPy arrays
X = np.array(image_data)
y = np.array(labels)

# Encode labels as numerical values
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model and label encoder
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(le, 'label_encoder.pkl')  # Save label encoder separately
print("Model and label encoder saved!")