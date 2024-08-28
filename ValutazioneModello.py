from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd

#Valutazione del modello
def evaluate_model(rf,nome_modello,X_train,X_test,y_train, y_test, y_pred):
    print(f"Report di classificazione {nome_modello}:\n", classification_report(y_test, y_pred, zero_division=1))
    print(f"Accuratezza {nome_modello}: ", accuracy_score(y_test,y_pred))

    #Grafico delle distribuzioni delle predizioni
    print("Distribuzione delle predizioni nel test set: ", Counter(y_pred))
    pred_counter= Counter(y_pred)
    labels= list(pred_counter.keys())
    counts= list(pred_counter.values())

    plt.figure(figsize=(10,6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Classi predette')
    plt.ylabel('Conteggio')
    plt.title("Distribuzione delle predizioni nel test set")
    plt.show()

    #Matrice di confusione
    cm= confusion_matrix(y_test, y_pred)
    disp= ConfusionMatrixDisplay(confusion_matrix= cm, display_labels= rf.classes_)
    disp.plot(cmap='Blues')
    disp.figure_.set_size_inches(12, 8)
    plt.show()

    #Cross validation
    scores= cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuratezza media della cross validation: ", scores.mean())

    print("Previsioni delle etichette Air_Quality_Category")
    print(y_pred)

    comparison= pd.DataFrame({'Vero':y_test, 'Previsto': y_pred})
    print("Confronto tra etichette vere e previste:")
    print(comparison)

    #Filtra le predizioni errate
    errors= comparison[comparison['Vero'] != comparison['Previsto']]
    print("Campioni con errore di previsione:")
    print(errors)
    errors_num= (y_test != y_pred).sum()
    print("Numero errori: ", errors_num)

    n_file= 'Confronto_previsioni_' + nome_modello +  '.csv'
    comparison.to_csv(n_file,index=False)
    print(f"Previsioni salvate in {n_file}")

