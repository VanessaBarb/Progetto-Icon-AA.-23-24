import RandomForestClassifier as rfc
import decisionTree as dect
import ValutazioneModello as evalmod
import randomForestBalanced as rfb
import Smote as sm
import Ontology as oy


def main():

    #Classificatore
    print("***Classificatore Random Forest***")
    df_rfc= rfc.load_data()
    df_rfc = rfc.prepare_data(df_rfc)
    X_train_rfc,X_test_rfc,y_train_rfc,y_test_rfc= rfc.prepare_model_data(df_rfc)
    rfc_model, scaler, X_train_res, y_train_res= rfc.train_model(X_train_rfc,X_test_rfc, y_train_rfc, y_test_rfc)
    rfc.evaluate_model(rfc_model,df_rfc, X_test_rfc, y_test_rfc,scaler, X_train_res, y_train_res)
    rfc.check_of(rfc_model, X_train_rfc, y_train_rfc, X_test_rfc, y_test_rfc, X_train_res, y_train_res)


    #Predittori: Decision-Tree e Random Forest

    print("\n\n***Decision-Tree***")
    #Decision-Tree
    df_dt= sm.smoteDataset()
    X_train_dt, X_test_dt, y_train_dt, y_test_dt= dect.prepare_data(df_dt)
    DecTree, y_pred_dt= dect.train_model(X_train_dt, y_train_dt, X_test_dt)
    #Valutazione decision-tree
    evalmod.evaluate_model(DecTree,"Decision-Tree", X_train_dt, X_test_dt, y_train_dt, y_test_dt, y_pred_dt)


    #RandomForest
    print("\n\n***Random Forest***")
    df_rfp = rfb.load_data()
    df_rfp_filtered= rfb.filter_data(df_rfp)
    X_train_rfp, X_test_rfp, y_train_rfp, y_test_rfp = rfb.prepare_data(df_rfp_filtered)
    y_pred_rfp, rf= rfb.train_model(X_train_rfp,X_test_rfp, y_train_rfp)
    #valutazione predittore Random Forest
    evalmod.evaluate_model(rf,"Random Forest", X_train_rfp,X_test_rfp,y_train_rfp,y_test_rfp, y_pred_rfp)



if __name__ == "__main__":
    main()

