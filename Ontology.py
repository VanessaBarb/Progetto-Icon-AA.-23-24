import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Carica il file CSV
df = pd.read_csv(r"C:\Users\Stefano\Desktop\ICON\ProgettoIcon\global_air_quality.csv")

# Crea un grafo RDF
g = Graph()

g.parse(r"C:\Users\Stefano\Desktop\Protege-5.6.3\Global_Aqi_Ontology.xml", format="xml")

# Definisci il namespace
NS = Namespace("http://www.semanticweb.org/ontologies/2024/7/Global-Air-Quality#")






def populate_ontology(df):
    df = df.sort_values(by=['Date'])
    #liste contenente il nome delle città, dei paesi e delle label
    city_list=[]
    country_list=[]
    label_list=[]
    #indice per numerare le misurazioni
    i=0
    for index, row in df.iterrows():
        i+=1
        padded_index = str(i).zfill(5)
        measurement_pm25_uri = URIRef(NS[f'Measurement_PM2_5_{padded_index}'])
        g.add((measurement_pm25_uri, RDF.type, NS['PM2_5']))
        g.add((measurement_pm25_uri, NS['value'], Literal(row['PM2.5'], datatype=XSD.float)))

        measurement_pm10_uri = URIRef(NS[f'Measurement_PM10_{padded_index}'])
        g.add((measurement_pm10_uri, RDF.type, NS['PM10']))
        g.add((measurement_pm10_uri, NS['value'], Literal(row['PM10'], datatype=XSD.float)))

        measurement_no2_uri = URIRef(NS[f'Measurement_NO2_{padded_index}'])
        g.add((measurement_no2_uri, RDF.type, NS['NO2']))
        g.add((measurement_no2_uri, NS['value'], Literal(row['NO2'], datatype=XSD.float)))

        measurement_so2_uri = URIRef(NS[f'Measurement_SO2_{padded_index}'])
        g.add((measurement_so2_uri, RDF.type, NS['SO2']))
        g.add((measurement_so2_uri, NS['value'], Literal(row['SO2'], datatype=XSD.float)))

        measurement_co_uri = URIRef(NS[f'Measurement_CO_{padded_index}'])
        g.add((measurement_co_uri, RDF.type, NS['CO']))
        g.add((measurement_co_uri, NS['value'], Literal(row['CO'], datatype=XSD.float)))

        measurement_o3_uri = URIRef(NS[f'Measurement_O3_{padded_index}'])
        g.add((measurement_o3_uri, RDF.type, NS['O3']))
        g.add((measurement_o3_uri, NS['value'], Literal(row['O3'], datatype=XSD.float)))

        temperature_uri = URIRef(NS[f'Temperature_{padded_index}'])
        g.add((temperature_uri, RDF.type, NS['Temperature']))
        g.add((temperature_uri, NS['value'], Literal(row['Temperature'], datatype=XSD.float)))

        humidity_uri = URIRef(NS[f'Humidity_{padded_index}'])
        g.add((humidity_uri, RDF.type, NS['Humidity']))
        g.add((humidity_uri, NS['value'], Literal(row['Humidity'], datatype=XSD.float)))

        wind_speed_uri = URIRef(NS[f'Wind_Speed_{padded_index}'])
        g.add((wind_speed_uri, RDF.type, NS['WindSpeed']))
        g.add((wind_speed_uri, NS['value'], Literal(row['Wind Speed'], datatype=XSD.float)))

        date_uri = URIRef(NS[f'Date_{padded_index}'])
        g.add((date_uri, RDF.type, NS['Date']))
        g.add((date_uri, NS['date'], Literal(row['Date']+"T00:00:00", datatype=XSD.dateTime)))

        aqi_uri = URIRef(NS[f'AQI_{padded_index}'])
        g.add((aqi_uri, RDF.type, NS['AQI']))
        g.add((aqi_uri, NS['value'], Literal(row['Air_Quality'], datatype=XSD.float)))


        country = row['Country']
        country_uri = URIRef(NS[country.replace(" ", "_")])
        if country not in country_list:
            country_list.append(country)
            g.add((country_uri, RDF.type, NS['Country']))



        city = row['City']
        city_uri = URIRef(NS[city.replace(" ", "_")])
        if city not in city_list:
            city_list.append(city)
            g.add((city_uri, RDF.type, NS['City']))
            g.add((city_uri, NS['Is_in'], country_uri))


        label = row['Air_Quality_Category']
        label_uri = URIRef(NS[label.replace(" ", "_")])
        if label not in label_list:
            label_list.append(label)
            g.add((label_uri, RDF.type, NS['AQI_Label']))



        g.add((measurement_pm25_uri, NS['Measured_in'], city_uri))
        g.add((measurement_pm10_uri, NS['Measured_in'], city_uri))
        g.add((measurement_no2_uri, NS['Measured_in'], city_uri))
        g.add((measurement_so2_uri, NS['Measured_in'], city_uri))
        g.add((measurement_co_uri, NS['Measured_in'], city_uri))
        g.add((measurement_o3_uri, NS['Measured_in'], city_uri))
        g.add((measurement_pm25_uri, NS['Measured_in'], city_uri))
        g.add((temperature_uri, NS['Measured_in'], city_uri))
        g.add((wind_speed_uri, NS['Measured_in'], city_uri))
        g.add((humidity_uri, NS['Measured_in'], city_uri))
        g.add((aqi_uri, NS['Measured_in'], city_uri))


        g.add((measurement_pm25_uri, NS['measured_on'], date_uri))
        g.add((measurement_pm10_uri, NS['measured_on'], date_uri))
        g.add((measurement_no2_uri, NS['measured_on'], date_uri))
        g.add((measurement_so2_uri, NS['measured_on'], date_uri))
        g.add((measurement_co_uri, NS['measured_on'], date_uri))
        g.add((measurement_o3_uri, NS['measured_on'], date_uri))
        g.add((measurement_pm25_uri, NS['measured_on'], date_uri))
        g.add((temperature_uri, NS['measured_on'], date_uri))
        g.add((wind_speed_uri, NS['measured_on'], date_uri))
        g.add((humidity_uri, NS['measured_on'], date_uri))
        g.add((aqi_uri, NS['measured_on'], date_uri))

        g.add((aqi_uri, NS['corresponds_to'], label_uri))

    # Serializza il grafo in formato XML
    g.serialize(destination=r"C:\Users\Stefano\Desktop\Protege-5.6.3\Global_Aqi_Ontology.owl", format='xml')



def max_pm25(g,ns):

    query= """
    PREFIX ns: <http://www.semanticweb.org/ontologies/2024/7/Global-Air-Quality#>
    SELECT ?City ?Date ?PM2_5
    WHERE {
    ?measurement ns:Measured_in ?City .
    ?measurement ns:value ?PM2_5 .
    ?measurement ns:measured_on ?Date .
    }
    ORDER BY DESC(?PM2_5)
    LIMIT 1
        """
    result=g.query(query)

    for row in result:
        print(row)
        print("Città con il PM2.5 più alto: {row.City},Data: {row.Date}, Valore PM2.5: {row.PM2_5}")









def analyze_ontology():
    for subj, pred, obj in g:
        print(subj, pred, obj)




#populate_ontology(df)
max_pm25(g,NS)