import decisionTree as dect
import ValutazioneModello as evalmod
import randomForestBalanced as rfb
import Ontology as oy
import pandas as pd

def main():



    #Predittori: Decision-Tree e Random Forest

    #Decision-Tree
    df= dect.load_data()
    df_dt= dect.balance_dataset(df)
    X_train_dt, X_test_dt, y_train_dt, y_test_dt= dect.prepare_data(df_dt)
    DecTree = dect.train_model(X_train_dt, X_test_dt, y_train_dt, y_test_dt)
    #Valutazione decision tree
    evalmod.evaluate_model(DecTree,X_train_dt, X_test_dt, y_train_dt, y_test_dt, "Decision Tree")

    #RandomForest
    df_rfp = rfb.load_data()
    df_rfp_filtered= rfb.filter_data(df_rfp)
    X_train_rfp, X_test_rfp, y_train_rfp, y_test_rfp = rfb.prepare_data(df_rfp_filtered)
    rf= rfb.train_model(X_train_rfp,X_test_rfp, y_train_rfp)
    #valutazione predittore Random Forest
    evalmod.evaluate_model(rf,X_train_rfp, X_test_rfp, y_train_rfp, y_test_rfp, "Random Forest")



if __name__ == "__main__":
    main()

