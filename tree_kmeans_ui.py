import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('injected _dataset.csv')

df = load_data()

# Preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])
df['day_of_week'] = df['DATE'].dt.dayofweek
df['hour_of_day'] = df['DATE'].dt.hour
label_encoder = LabelEncoder()
df['AGENT_encoded'] = label_encoder.fit_transform(df['AGENT'])
df['DATE_numeric'] = df['DATE'].astype(np.int64)
X = df[['DATE_numeric', 'AGENT_encoded', 'day_of_week', 'hour_of_day']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar for hyperparameters
st.sidebar.header('Model Hyperparameters')
nu = st.sidebar.slider('nu (One-Class SVM)', min_value=0.01, max_value=1.0, value=0.07)

# Split data
X_train, X_test = train_test_split(X_scaled, test_size=0.25, random_state=42)

# One-Class SVM model
svm_model = OneClassSVM(kernel='rbf', nu=nu)
svm_model.fit(X_train)

# Decision Tree model
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, np.where(svm_model.predict(X_train) == 1, 0, 1))

# Decision Tree visualization
st.header('Decision Tree Visualization')
fig, ax = plt.subplots(figsize=(15, 20))  
plot_tree(tree_classifier, filled=True, feature_names=X.columns, class_names=["Inlier", "Outlier"], max_depth=4, ax=ax)
st.pyplot(fig)

# One-Class SVM results
st.header('One-Class SVM Results')
# Completeness and Robustness metrics
completeness = (svm_model.predict(X_train) == 1).sum() / len(X_train)
st.write(f"Completeness: {completeness}")
# Robustness
noise_level = 0.3
np.random.seed(42)
noise = np.random.normal(0, noise_level, X_test.shape)
perturbed_X_test = np.copy(X_test)
perturbed_X_test += noise
perturbed_test_predictions = svm_model.predict(perturbed_X_test)
robustness = np.mean(svm_model.predict(X_test) == perturbed_test_predictions)
st.write(f"Robustness: {robustness}")

# Decision Tree results
st.header('Decision Tree Results')
# Metrics
tree_train_predictions = tree_classifier.predict(X_train)
tree_test_predictions = tree_classifier.predict(X_test)
accuracy = metrics.accuracy_score(np.where(svm_model.predict(X_test) == 1, 0, 1), tree_test_predictions)
precision = metrics.precision_score(np.where(svm_model.predict(X_test) == 1, 0, 1), tree_test_predictions)
recall = metrics.recall_score(np.where(svm_model.predict(X_test) == 1, 0, 1), tree_test_predictions)
f1_score = metrics.f1_score(np.where(svm_model.predict(X_test) == 1, 0, 1), tree_test_predictions)
st.write(f"Accuracy: {accuracy}")
st.write(f"Precision: {precision}")
st.write(f"Recall: {recall}")
st.write(f"F1-Score: {f1_score}")

# Load dataset for KMeans
#filepath = os.getcwd() + "\src\Datasets"  # Store file path to data in variable
dataset = pd.read_csv('injected _dataset.csv')  # Store data in dataframe object

# Vectorize text data
vectorizer = TfidfVectorizer(stop_words='english')  # Removes basic English words, can be improved on
training_df = vectorizer.fit_transform(dataset['SUMMARY'])

# Sidebar for KMeans hyperparameters
st.sidebar.header('KMeans Hyperparameters')
k = st.sidebar.slider('Number of clusters (k)', min_value=2, max_value=50, value=5)

# Create KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)  # Random state ensures reproducibility

# Fit model
dataset['CLUSTER'] = kmeans.fit_predict(training_df)

st.header('K-Means Clustering')
# Display silhouette score
silhouette_avg = silhouette_score(training_df, kmeans.labels_)
st.write(f"Silhouette Score on Test Data: {silhouette_avg}")

# Display number of unique clusters
n_clusters = dataset['CLUSTER'].nunique()
st.write(f"Number of unique clusters: {n_clusters}")

# Display number of objects in each cluster
cluster_counts = dataset['CLUSTER'].value_counts()
st.write("Number of objects in each cluster:")
st.write(cluster_counts)

# Visualize the clusters using a scatter plot
st.header('Clustering Visualization')
st.write("Visualizing clusters using scatter plot:")
fig, ax = plt.subplots()
scatter = ax.scatter(dataset.values[:, 0], dataset.values[:, 4], c=kmeans.labels_, cmap='viridis', alpha=0.8)
ax.set_title('Clustering of Test Data')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Display cluster centers
cluster_centers = kmeans.cluster_centers_
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, linewidths=3, color='r', label='Cluster Centers')
ax.legend()

# Show plot in Streamlit
st.pyplot(fig)

