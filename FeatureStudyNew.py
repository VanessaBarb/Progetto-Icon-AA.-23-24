import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df= pd.read_csv(r"globalAirNew.csv")
print(df.head())

print(df.isnull().sum())

print(df.describe())

#Distribuzione delle feature PM2.5 e PM10
for column in [ 'PM2.5', 'PM10']:
    plt.figure(figsize=(10,6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Distribuzione di {column}')
    plt.xlabel(f'{column}')
    plt.ylabel('Frequenza')
    plt.show()


#Correlazioni tra feature
correlation_matrix= df[['SO2', 'O3', 'PM2.5','PM10', 'Temperature', 'Humidity', 'Wind Speed', 'Air_Quality']].corr()
sns.heatmap(correlation_matrix, annot= True, cmap= 'coolwarm')
plt.title('Matrice di Correlazione')
plt.show()

#Distribuzione delle categorie dell'aqi
print(df['Air_Quality_Category'].value_counts())


# Box plot dei livelli di PM2.5 per città
plt.figure(figsize=(15, 8))
sns.boxplot(x='City', y='PM2.5', data=df)
plt.title('Livelli di PM2.5 per città')
plt.xlabel('Città')
plt.ylabel('PM2.5')
plt.xticks(rotation=90)
plt.show()


#raggruppamento per città e calcolo della media degli inquinanti
city_pm25_mean= df.groupby('City')[['PM2.5']].mean()
#Media dei livelli di PM2.5 per città
plt.figure(figsize=(15, 8))
city_pm25_mean['PM2.5'].plot(kind='bar')
plt.title('Media dei livelli di PM2.5 per città')
plt.xlabel('Cittò')
plt.ylabel('Media PM2.5')
plt.xticks(rotation=90)
plt.show()
"""
#Analisi di box plot per categorie
for column in ['SO2','O3','PM2.5','PM10', 'Temperature','Humidity','Wind Speed']:
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Air_Quality_Category', y=column, data=df)
    plt.title(f"Box plot di {column} per categoria di qualità dell'aria")
    plt.show()
"""

"""
for column in ['SO2', 'O3', 'PM2.5', 'PM10']:
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=column, hue='City', multiple='stack', binwidth=1)
    plt.title(f'Histogramma di {column} per città')
    plt.show()
"""
