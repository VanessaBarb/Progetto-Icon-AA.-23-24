import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def preprocess_data(df):
    #Conversione numerica
    df['Is_Stagnant'] = pd.to_numeric(df['Is_Stagnant'], errors='coerce').fillna(0).astype(int)
    df['HasRained'] = pd.to_numeric(df['HasRained'], errors='coerce').fillna(0).astype(int)
    X = df[['Dispersion_Index', 'Is_Stagnant', 'HasRained']]
    y = df['Air_Quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def grid_search(X_train, y_train):
    #Ricerca degli iperparametri
    param_grid = {
        'n_neighbors': [5, 10, 20, 30, 40],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'p': [1, 2]
    }
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Migliori parametri trovati:", grid_search.best_params_)
    return grid_search.best_estimator_

def trainModel(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def learn_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Train MSE')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation MSE')
    plt.title('Curva di Apprendimento')
    plt.xlabel('Numero di Esempi di Addestramento')
    plt.ylabel('Errore Quadratico Medio (MSE)')
    plt.legend()
    plt.grid()
    plt.show()

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_cv= -np.mean(cv_scores)
    print(f"Mean squared error(CV): {mse_cv:.2f}")

def evalModel(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

"""
X, y = preprocess_data(df)

best_knn = grid_search(X_train, y_train)
learn_curve(best_knn, X, y)

y_pred = best_knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

"""

