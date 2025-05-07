# Task 8 - Clustering with K-Means
---

## 🔥 What I Did
- Loaded the dataset and selected two features (`sepal_length`, `sepal_width`) for easy visualization.
- Standardized the features using **StandardScaler**.
- Used the **Elbow Method** to find the optimal number of clusters.
- Applied **K-Means Clustering** with the optimal `K`.
- Visualized the clusters after reducing dimensions using **PCA**.
- Evaluated clustering performance using the **Silhouette Score**.

---

## 🛠 Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

---


## 📊 Results
| Metric | Value |
|--------|-------|
| Silhouette Score | ~0.43 (on Iris dataset with 3 clusters) |
![image](https://github.com/user-attachments/assets/3916c06c-61a3-4667-9691-137a3ed72987)


---

## 📷 Visualization
> Clusters Visualized using PCA.

![image](https://github.com/user-attachments/assets/90999e22-3d5b-4c3c-b61f-148e19a34044)
![image](https://github.com/user-attachments/assets/7c1bf6f6-bbf0-4a44-86c6-3bbcd1f651ab)


---

## 🔗 Dataset Source
- [Iris Dataset - Seaborn Sample](https://github.com/mwaskom/seaborn-data/blob/master/iris.csv)
