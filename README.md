
# 🎬 Movie Recommendation System using Neural Networks

This project showcases a **Movie Recommendation System** built using deep learning techniques. The model is designed to predict movie ratings based on user preferences, leveraging embedding layers for users and movies in a collaborative filtering setup.

## 📌 Project Overview

Traditional recommendation systems rely on matrix factorization or content-based filtering. In this project, we use **neural networks** to learn latent features of users and movies from historical rating data. The model is trained to predict how a user would rate a specific movie, given embeddings that capture complex patterns in the data.

## 🧠 Technologies & Skills Used

- **Python** – Primary language for coding and data handling
- **NumPy & Pandas** – Data manipulation and analysis
- **TensorFlow / Keras** – Deep learning framework used to build the recommendation model
- **Embedding Layers** – Represent users and movies as dense vectors
- **Train/Test Split** – To evaluate the model’s generalization
- **Mean Squared Error (MSE)** – Loss function for optimization

## 🛠️ How it Works

1. Load and preprocess user-movie-rating data.
2. Create embedding layers for users and movies.
3. Build a neural network model combining embeddings and dense layers.
4. Train the model on the training set.
5. Evaluate performance using test data.

## 📂 File

- `Movie_Recommender_Cleaned.ipynb`: The Jupyter notebook containing the full implementation of the recommendation system.

## 🚀 Future Improvements

- Add dropout layers to reduce overfitting.
- Experiment with different optimizers and deeper architectures.
- Add movie metadata (genre, year, etc.) for hybrid recommendation.

---

This project demonstrates a practical and modern approach to building intelligent recommendation systems using deep learning.
