from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# check the number of different classes in y
print('Number of different classes in y: ', len(np.unique(y)))

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to standardize the test data
X_test_scaled = scaler.transform(X_test)

# Perform PCA to reduce the dimensionality of the standardized training data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Apply the same PCA transformation to the standardized test data
X_test_pca = pca.transform(X_test_scaled)

# Create a k-NN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the PCA-transformed training data
knn.fit(X_train_pca, y_train)

# Predict the test set
y_pred = knn.predict(X_test_pca)

# Print the accuracy
print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_pred))

# Plot the decision boundary
h = .02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, alpha=0.1)
colors = ['navy', 'turquoise', 'darkorange'] # change in accordance with number of classes
lw = 2
for color, i, target_name in zip(colors, np.arange(len(np.unique(y))), iris.target_names):
    plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS Training Dataset')
plt.show()

# Visualize the PCA-transformed training data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, np.arange(len(np.unique(y))), iris.target_names):
    plt.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS Training Dataset')

# Visualize the PCA-transformed test data
plt.subplot(1, 2, 2)
for color, i, target_name in zip(colors, np.arange(len(np.unique(y))), iris.target_names):
    plt.scatter(X_test_pca[y_test == i, 0], X_test_pca[y_test == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS Testing Dataset')
plt.show()
