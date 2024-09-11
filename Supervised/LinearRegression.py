
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix(df):
    corr = df[['Dispersion_Index', 'Air_Quality', 'HasRained', 'Is_Stagnant']].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Matrice di correlazione tra Dispersion Index e AQI")
    plt.show()

def plot_scatter(df):
    plt.figure(figsize=(6, 4))
    plt.scatter(df['Dispersion_Index'], df['Air_Quality'], alpha=0.5)
    plt.title('Dispersion Index vs AQI')
    plt.xlabel('Dispersion Index')
    plt.ylabel('AQI')
    plt.show()

def preprocess_data(df):
    X = df[['Dispersion_Index', 'Is_Stagnant']]
    y = df['Air_Quality']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def learn_curve(X, y):
    """Visualizza la curva di apprendimento del modello di regressione lineare."""
    train_sizes, train_scores, val_scores = learning_curve(
        LinearRegression(), X, y, cv=5,
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

def train_and_evaluate(X, y):
    lr = LinearRegression()

    # Cross-Validation
    cv_scores = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_cv = -np.mean(cv_scores)
    print(f"Mean Squared Error (CV): {mse_cv:.2f}")

    # Valutazione del modello sui dati di test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (Test): {mse:.2f}")
    print(f"R-squared (Test): {r2:.2f}")

    return y_test, y_pred

def plot_results(y_test, y_pred):
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('True AQI')
    plt.ylabel('Predicted AQI')
    plt.title('True AQI vs Predicted AQI')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Linea di uguaglianza
    plt.show()



"""
correlation_matrix(df)
plot_scatter(df)

X, y = preprocess_data(df)
learn_curve(X, y)
y_test, y_pred = train_and_evaluate(X, y)

plot_results(y_test, y_pred)
"""
