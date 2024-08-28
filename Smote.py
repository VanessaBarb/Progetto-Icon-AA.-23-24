import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def smoteDataset():
    df= pd.read_csv("globalAirNew.csv")

    features=['PM2.5','PM10']
    X= df[features]
    y= df['Air_Quality_Category']

    smote= SMOTE(random_state=42)
    X_res, y_res= smote.fit_resample(X,y)

    #Creazione di un nuovo dataframe con dati bilanciati
    df_resampled= pd.concat([pd.DataFrame(X_res, columns= features), pd.DataFrame(y_res, columns=['Air_Quality_Category'])], axis=1)

    df_resampled.to_csv("globalAirNew_balanced.csv", index= False)
    print("Done..")
    return df_resampled