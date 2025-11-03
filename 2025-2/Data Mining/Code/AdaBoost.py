import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from types import SimpleNamespace

weak_learner = DecisionTreeClassifier(max_depth=1, random_state=0)

def adaboost(D, K):
    """
    D : dataset with attributes D.X y D.y
    K : Number of weak classifiers
    """

    X, y = D.X, D.y
    n = len(X)
    w = np.ones(n) / n  # uniform initial weights

    for t in range(K):

        # 1️⃣ Weighted sampling of indices with replacement
        idx = np.random.choice(n, size=n, replace=True, p=w)
        X_t, y_t = X[idx], y[idx]

        # 2️⃣ Train weak learner on the weighted sample
        M_t = weak_learner.fit(X_t, y_t)

        # 3️⃣ Predict on the full dataset
        y_pred = M_t.predict(X)

        # 4️⃣ Compute weighted classification error
        epsilon_t = np.sum(w * (y_pred != y))

        # 5️⃣ Stop if classifier is perfect
        if epsilon_t == 0:
            break

        # 6️⃣ Compute classifier weight (alpha)
        alpha_t = np.log((1 - epsilon_t) / epsilon_t)

        # 7️⃣ Update instance weights: increase for misclassified samples
        w = np.where(y_pred == y, w, w * np.exp(alpha_t))

        # 8️⃣ Normalize weights to sum to 1
        w = w / np.sum(w)

    # 9️ Return classifier, its weight, error, and final sample weights
    return M_t, alpha_t, epsilon_t, w


# ======= Example dataset =======
X, y = make_classification(n_samples=8, n_features=2, n_informative=2,
                           n_redundant=0, random_state=42)
y = np.where(y == 0, -1, 1)  # etiquetas {-1, 1}

# Create dataset object compatible with .X and .y
D = SimpleNamespace(X=X, y=y)

# Run AdaBoost for one iteration
M, alpha, epsilon, w_new = adaboost(D, 1)

print("Weighted error ε:", round(epsilon, 4))
print("Classifier weight α:", round(alpha, 4))
print("New sample weights w_i:", np.round(w_new, 4))
