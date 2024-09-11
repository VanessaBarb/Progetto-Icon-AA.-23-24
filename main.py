import CalculateAqi as caqi
import FeatureStudy as fs
from Supervised import init
from Supervised import KNNRegressor as kn
from Supervised import LinearRegression as lr
from Supervised import RandomForest as rf
from Supervised import ValutazioneModello as evm
from Unsupervised import KMeans as kms
from Bayesian import BayesianNetwork as bn
from KnowledgeBase import KB as kb
import pandas as pd


def main():
    #Calcolo dell'AQI e creazione del nuovo dataset
    print("***Calcolo dell'AQI e generazione del nuovo dataset globalAir00.csv***")
    original_df= pd.read_csv(r"globalAir.csv")
    df_AQI= caqi.calculate_And_Save(original_df)
    caqi.AQI_Distribution(df_AQI)

    print("\n\n***Studio delle Feature***")
    fs.feature_correlation(df_AQI)
    fs.PM_distribution(df_AQI)
    fs.category_distribution(df_AQI)
    #Crea un nuovo dataset contentente le medie
    df_New = fs.calc_avg(df_AQI)

    print("\n\n***Creazione e gestione della knowledge base...****")
    kb.create_kb(df_New)
    kb.rule_definition()
    kb.create_csv(r"globalAir01.csv","Final_globalAir.csv")

    df_final = pd.read_csv(r"Final_globalAir.csv")

    print("\n\n***Random Forest***")
    df_rf_cleaned = rf.preprocessing(df_final)
    X_top_rf, y_rf = rf.feature_imp_matrix_corr(df_rf_cleaned)
    X_rf_resampled, y_rf_resampled= rf.smote_good(X_top_rf, y_rf)
    randomForest, X_train_rf, X_test_rf, y_train_rf, y_test_rf = rf.train_model(X_rf_resampled,y_rf_resampled)
    evm.evaluate_model(randomForest, X_train_rf, X_test_rf, y_train_rf, y_test_rf, "Random Forest")

    print("\n\n***Linear Regression***")
    lr.correlation_matrix(df_final)
    lr.plot_scatter(df_final)
    X_lr, y_lr = lr.preprocess_data(df_final)
    lr.learn_curve(X_lr, y_lr)
    y_test_lr, y_pred_lr = lr.train_and_evaluate(X_lr, y_lr)
    lr.plot_results(y_test_lr, y_pred_lr)

    print("\n\n**K-neighbors Regression")
    X_kn, y_kn = kn.preprocess_data(df_final)
    X_train_kn, X_test_kn, y_train_kn, y_test_kn = kn.trainModel(X_kn, y_kn)
    best_knn = kn.grid_search(X_train_kn, y_train_kn)
    kn.learn_curve(best_knn, X_kn, y_kn)
    y_pred_kn = best_knn.predict(X_test_kn)
    kn.evalModel(y_test_kn, y_pred_kn)

    print("\n\n***K-means Clustering***")
    df_out = kms.manage_outliers()
    df_scaled = kms.scale_values(df_out)
    kms.find_k(df_scaled)
    oc = 4
    kms.K_means_clustering(oc, df_scaled)

    print("\n\n***Bayesian Network***")
    bn.create_discretizer()
    model = bn.load_model()
    bn.visualize_model(model)
    bn.bayesian_Infer(21.63,131.64,91.93,25.92,7.8,39.01,-2.21,72.86,4.64,False,False,2.68, model)




if __name__ == "__main__":
    main()

