import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the processed data
data = pd.read_pickle("../../data/interim/03_data_processed.pkl")

# Verify the categories
print("Unique categories in the dataset:", data['category'].unique())

# Check the count of each category
print("Count of each category:\n", data['category'].value_counts())

# Prepare the data for training
X = data[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]]
y = data["label"]

# Handle missing values by imputing them
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Use StratifiedKFold to ensure each category is represented in both train and test sets
strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = next(strat_kf.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Print counts of each label in training and testing sets
print("Training set label counts:\n", y_train.value_counts())
print("Testing set label counts:\n", y_test.value_counts())

# Oversample the training set using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print counts of each label in the resampled training set
print("Resampled training set label counts:\n", y_train_resampled.value_counts())

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Adding a Neural Network model
def build_neural_network(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes, adjust accordingly

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Convert labels to one-hot encoding for neural network
y_train_nn = to_categorical(y_train_resampled)
y_test_nn = to_categorical(y_test)

nn_model = build_neural_network(X_train_resampled.shape[1])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the Neural Network
history = nn_model.fit(X_train_resampled, y_train_nn,
                       validation_split=0.2,
                       epochs=100,
                       batch_size=32,
                       callbacks=[early_stopping],
                       verbose=2)

# Evaluate the Neural Network
y_pred_nn = nn_model.predict(X_test)
y_pred_nn_classes = y_pred_nn.argmax(axis=-1)

accuracy_nn = accuracy_score(y_test, y_pred_nn_classes)
print("\nNeural Network Classification Report:\n")
print(classification_report(y_test, y_pred_nn_classes))
print(f"Neural Network Accuracy: {accuracy_nn:.4f}")

# Compute confusion matrix for Neural Network
conf_mat_nn = confusion_matrix(y_test, y_pred_nn_classes)
confusion_matrices['Neural Network'] = conf_mat_nn

# Train and evaluate other models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Compute confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    confusion_matrices[model_name] = conf_mat

# Plot confusion matrices
plt.figure(figsize=(20, 8))

for i, (model_name, conf_mat) in enumerate(confusion_matrices.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')

# Save confusion matrices as an image
plt.savefig('confusion_matrices.png')

# Show the plot
plt.show()

