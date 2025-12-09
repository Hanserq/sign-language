import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'

data = []
labels = []

# 1. Load all the data files
print("Loading data...")
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pickle"):
        file_path = os.path.join(DATA_DIR, filename)
        
        with open(file_path, 'rb') as f:
            dict_data = pickle.load(f)
            
        # Add this file's data to our big list
        data.extend(dict_data['data'])
        labels.extend(dict_data['labels'])

# Convert to numpy arrays for the AI to understand
data = np.array(data)
labels = np.array(labels)

# 2. Split into Training and Test sets
# We use 80% of data to teach, and 20% to test if it learned correctly
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# 3. Train the Model
print("Training the model... (This might take a few seconds)")
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 4. Test Accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"✅ Model Trained! Accuracy: {score * 100:.2f}%")

# 5. Save the trained model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
print("💾 Model saved as 'model.p'")
