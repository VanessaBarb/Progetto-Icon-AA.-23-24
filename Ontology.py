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
    #dizionari contenenti il nome delle città, dei paesi e delle label
    city_dict = {}
    country_dict = {}
    label_dict = {}
    #indice per numerare le misurazioni
    i=0
    #indici per numerare città,paesi e label
    cr=0
    ci=0
    l=0
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


         # Gestione dei paesi
        country = row['Country']
        if country not in country_dict:
            cr += 1
            padded_index_cr = str(cr).zfill(5)
            country_uri = URIRef(NS[f'Country_{padded_index_cr}'])
            country_dict[country] = country_uri
            g.add((country_uri, RDF.type, NS['Country']))
            g.add((country_uri, NS['name'], Literal(country, datatype=XSD.string)))
        else:
            country_uri = country_dict[country]

        # Gestione delle città
        city = row['City']
        if city not in city_dict:
            ci += 1
            padded_index_ci = str(ci).zfill(5)
            city_uri = URIRef(NS[f'City_{padded_index_ci}'])
            city_dict[city] = city_uri
            g.add((city_uri, RDF.type, NS['City']))
            g.add((city_uri, NS['name'], Literal(city, datatype=XSD.string)))
            g.add((city_uri, NS['Is_in'], country_uri))
        else:
            city_uri = city_dict[city]

        # Gestione delle label
        label = row['Air_Quality_Category']
        if label not in label_dict:
            l += 1
            padded_index_l = str(l).zfill(5)
            label_uri = URIRef(NS[f'Label_{padded_index_l}'])
            label_dict[label] = label_uri
            g.add((label_uri, RDF.type, NS['AQI_Label']))
            g.add((label_uri, NS['label'], Literal(label, datatype=XSD.string)))
        else:
            label_uri = label_dict[label]




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
    g.serialize(destination=r"C:\Users\Stefano\Desktop\Protege-5.6.3\Global_Aqi_Ontology.xml", format='xml')


