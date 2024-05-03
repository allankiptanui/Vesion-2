from model import NeuralNetwork
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load training and testing data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [list(line.strip()) for line in data]

def encode_labels(labels):
    label_map = {'#': [1, 0, 0, 0], '.': [0, 1, 0, 0], 'o': [0, 0, 1, 0], '@': [0, 0, 0, 1]}
    return [label_map[label] for label in labels]

def preprocess_data(train_file, test_file):
    # Load training data
    train_data = load_data(train_file)
    train_labels = encode_labels([pixel for line in train_data for pixel in line])
    train_inputs = np.array(train_labels)

    # Load testing data
    test_data = load_data(test_file)
    test_labels = encode_labels([pixel for line in test_data for pixel in line])
    test_inputs = np.array(test_labels)

    return train_inputs, test_inputs, np.array(train_labels), np.array(test_labels)

def main():
    # Data preprocessing
    train_file = 'HW3_Training-1.txt'
    test_file = 'HW3_Testing-1.txt'
    X_train, X_test, y_train, y_test = preprocess_data(train_file, test_file)

    # Define configurations to test
    configurations = [
        {'hidden_size': 16, 'learning_rate': 0.1, 'activation': 'sigmoid', 'weight_scale': 0.01},
        {'hidden_size': 16, 'learning_rate': 0.01, 'activation': 'bipolar_sigmoid', 'weight_scale': 0.1},
        {'hidden_size': 16, 'learning_rate': 0.1, 'activation': 'sigmoid', 'weight_scale': 0.001}
    ]

    for config in configurations:
        print("Testing configuration:", config)
        # Initialize neural network with configuration
        nn = NeuralNetwork(input_size=len(X_train[0]), output_size=len(y_train[0]), **config)

        # Train the model
        nn.train(X_train, y_train, epochs=100)

        # Test the model
        predictions = nn.predict(X_test)
        predicted_labels = [np.argmax(prediction) for prediction in predictions]

        # Calculate evaluation metrics
        accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_labels)
        precision = precision_score(np.argmax(y_test, axis=1), predicted_labels, average='weighted')
        recall = recall_score(np.argmax(y_test, axis=1), predicted_labels, average='weighted')
        f1 = f1_score(np.argmax(y_test, axis=1), predicted_labels, average='weighted')

        # Print evaluation metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)

if __name__ == "__main__":
    main()
