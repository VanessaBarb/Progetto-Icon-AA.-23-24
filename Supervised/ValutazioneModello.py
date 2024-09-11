
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import  learning_curve, cross_val_score
import numpy as np

#Valutazione del modello
def evaluate_model(model,X_train,X_test,y_train, y_test, model_name):

    #Fai predizioni
    y_train_pred= model.predict(X_train)
    y_test_pred= model.predict(X_test)

    print(f"\n***Metriche per il modello {model_name}:***")
    print("Accuratezza del training: ", accuracy_score(y_train, y_train_pred))
    print("Accuratezza del test: ", accuracy_score(y_test, y_test_pred))
    print("Precisione: ", precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
    print("Recall: ", recall_score(y_test, y_test_pred, average= 'weighted', zero_division=0))
    print("F1-score: ", f1_score(y_test, y_test_pred, average='weighted',zero_division= 0))
    print("Matrice di confusione:\n", confusion_matrix(y_test, y_test_pred))
    print("Report di classificazione:\n", classification_report(y_test, y_test_pred, zero_division=0))

    #cross-validation per verificare la robustezza del modello
    cv_scores= cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n{model_name} Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    #Curva di apprendimmento
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5,
        n_jobs=-1, train_sizes= np.linspace(0.1,1.0,10), scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, train_mean,'o-', color='r', label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color='g',label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.title(f"Curva di apprendimento per {model_name}")
    plt.xlabel("Dimensione del Training set")
    plt.ylabel("Accuratezza")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
