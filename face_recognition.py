import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(f"Number of classes: {n_classes}")
print(f"Images shape: {lfw_people.images.shape}")
print(f"Number of samples: {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

n_components = 150
print(f"Extracting the top {n_components} eigenfaces from {X_train.shape[0]} faces")
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

svc = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='auto')
svc.fit(X_train_pca, y_train)
y_pred = svc.predict(X_test_pca)

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

fig, ax = plt.subplots(2, 5, figsize=(15, 8))
for i in range(10):
    ax[i // 5, i % 5].imshow(X_test[i].reshape(50, 37), cmap='gray')
    ax[i // 5, i % 5].set_title(f'True: {target_names[y_test[i]]}\nPred: {target_names[y_pred[i]]}')
    ax[i // 5, i % 5].axis('off')
plt.show()
