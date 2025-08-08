import cv2
import numpy as np
import os

def extract_color_histogram(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")
    img = cv2.resize(img, (100, 100))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8,8,8], [0,256, 0,256, 0,256])
    return cv2.normalize(hist, hist).flatten()

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

def load_dataset(folder):
    features = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        features.append(extract_color_histogram(path))
    return np.array(features)

# Load training data
dry_features = load_dataset("dry")
wet_features = load_dataset("wet")

# Combine data and labels
X = np.vstack((dry_features, wet_features))
y = np.array([0]*len(dry_features) + [1]*len(wet_features))  # 0: dry, 1: wet

# Test and classify
test_folder = "test"

for file in os.listdir(test_folder):
    test_feat = extract_color_histogram(os.path.join(test_folder, file))
    distances = [chi2_distance(test_feat, train_feat) for train_feat in X]
    prediction = y[np.argmin(distances)]
    label = "Dry Waste" if prediction == 0 else "Wet Waste"
    print(f"{file} --> {label} | Distance: {min(distances):.4f}")
