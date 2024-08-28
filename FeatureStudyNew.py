import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




#Distribuzione delle feature PM2.5 e PM10
def PM_distribution(df):
    for column in [ 'PM2.5', 'PM10']:
        plt.figure(figsize=(10,6))
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f'Distribuzione di {column}')
        plt.xlabel(f'{column}')
        plt.ylabel('Frequenza')
        plt.show()


#Correlazioni tra feature
def feature_correlation(df):
    correlation_matrix= df[['SO2', 'O3', 'PM2.5','PM10', 'Temperature', 'Humidity', 'Wind Speed', 'Air_Quality']].corr()
    sns.heatmap(correlation_matrix, annot= True, cmap= 'coolwarm')
    plt.title('Matrice di Correlazione')
    plt.show()


#Distribuzione delle categorie dell'aqi
def category_distribution(df):
    counts = df['Air_Quality_Category'].value_counts()

    # Visualizza un grafico a torta
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Category distribution')
    plt.show()


# Box plot dei livelli di PM2.5 per città
def PM2_5_per_city(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='City', y='PM2.5', data=df)
    plt.title('Livelli di PM2.5 per città')
    plt.xlabel('Città')
    plt.ylabel('PM2.5')
    plt.xticks(rotation=90)
    plt.show()


#raggruppamento per città e calcolo della media degli inquinanti
def avg_PM2_5(df):
    city_pm25_mean= df.groupby('City')[['PM2.5']].mean()
    #Media dei livelli di PM2.5 per città
    plt.figure(figsize=(10, 6))
    city_pm25_mean['PM2.5'].plot(kind='bar')
    plt.title('Media dei livelli di PM2.5 per città')
    plt.xlabel('Cittò')
    plt.ylabel('Media PM2.5')
    plt.xticks(rotation=90)
    plt.show()



df= pd.read_csv(r"globalAirNew.csv")


feature_correlation(df)
PM_distribution(df)
category_distribution(df)
PM2_5_per_city(df)
avg_PM2_5(df)
