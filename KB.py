from pyswip import Prolog
import pandas as pd
import csv


def create_kb(dataset):
    df = pd.read_csv(dataset)

    # Dizionari per raccogliere fatti separati per predicato
    geography_facts = []
    pollutants_facts = []
    climate_facts = []
    monthly_averages_facts = []
    AQI_facts = []

    #dizionari per gestire le eventuali ripetizioni
    geo_dict = []
    avg_dict = {}


    for _, row in df.iterrows():
        #controlli per evitare la ripetizione del fatto
        if row['City'] not in geo_dict:
            geo_dict.append(row['City'])
            geography_facts.append(f"geography('{row['City']}', '{row['Country']}').")

        if row['City'] not in avg_dict:
            avg_dict[row['City']] = set()

        if row['Month'] not in avg_dict[row['City']]:
            avg_dict[row['City']].add(row['Month'])
            monthly_averages_facts.append(
                f"monthly_averages('{row['City']}', {int(row['Month'])}, "
                f"{float(row['Monthly_Avg_Temperature'])}, {float(row['Monthly_Avg_Wind_Speed'])})."
            )


        pollutants_facts.append(
            f"pollutants('{row['City']}', {int(row['Month'])}, {int(row['Day'])}, "
            f"{float(row['PM2.5'])}, {float(row['PM10'])}, {float(row['NO2'])}, "
            f"{float(row['SO2'])}, {float(row['O3'])}, {float(row['CO'])})."
        )


        climate_facts.append(
            f"climate_factors('{row['City']}', {int(row['Month'])}, {int(row['Day'])}, "
            f"{float(row['Temperature'])}, {float(row['Humidity'])}, {float(row['Wind Speed'])})."
        )

        AQI_facts.append(
            f"aqi('{row['City']}', {int(row['Month'])}, {int(row['Day'])}, "
            f"{row['Air_Quality']}, '{row['Air_Quality_Category']}')."
        )

    with open('knowledge_base.pl', 'w') as file:
        # Scrivi tutti i fatti raggruppati per predicato
        file.write("\n% Fatti geografici\n")
        file.write("\n".join(geography_facts) + "\n")

        file.write("\n% Fatti sugli inquinanti\n")
        file.write("\n".join(pollutants_facts) + "\n")

        file.write("\n% Fatti sui fattori climatici\n")
        file.write("\n".join(climate_facts) + "\n")

        file.write("\n% Fatti sulle medie mensili\n")
        file.write("\n".join(monthly_averages_facts) + "\n")

        file.write("\n% Fatti sulla qualità dell'aria \n")
        file.write("\n".join(AQI_facts) + "\n")

    print("KB creata")

#metodo per la definizione delle regole della KB
def rule_definition():
    with open('knowledge_base.pl', 'a') as file:
        rule = """
            %restituzioni delle misurazioni per una città in un giorno
            get_measurements(City, Month, Day, PM2_5, PM10, NO2, SO2, O3, CO, Temperature, Humidity, Wind_Speed, AQI_value, AQI_category) :-
            pollutants(City, Month, Day, PM2_5, PM10, NO2, SO2, O3, CO),
            climate_factors(City, Month, Day, Temperature, Humidity, Wind_Speed),
            aqi(City, Month, Day, AQI_value, AQI_category).
            
            
            % Trova la città con l AQI_value più alto
            highest_aqi(City, Month, Day, Max_AQI) :-
            findall(AQI_value, aqi(_, _, _, AQI_value, _), AQI_list),  
            max_list(AQI_list, Max_AQI),                               
            aqi(City, Month, Day, Max_AQI, _).   
        
        
            %verifica se ha piovuto
            rainy_day(City, Month, Day,HasRained):-
            climate_factors(City, Month, Day, Temperature, Humidity, Wind_Speed),
            monthly_averages(City, Month, Average_Temperature,Average_Wind_Speed),
            (   Temperature < Average_Temperature,
            Wind_Speed > Average_Wind_Speed,
            Humidity > 90
        ->  HasRained = true
        ;   HasRained = false
        ).
        
        
         %verifica della stagnazione dell aria
            stagnant_air(City, Month, Day,Is_Stagnant):-
            climate_factors(City, Month, Day, Temperature, Humidity, Wind_Speed),
            (   Temperature > 28,
            Wind_Speed < 3.5,
            Humidity > 70
        ->  Is_Stagnant = true
        ;   Is_Stagnant = false
        ).
        
        
        % Calcolo dell indice di dispersione 
        pollutant_dispersion(City, Month, Day, Dispersion_Index) :-
        climate_factors(City, Month, Day, _, Humidity, Wind_Speed),
        Dispersion_Index is Wind_Speed * (1 / (1 + (Humidity / 100))).
        """

        file.write(rule)
        print("Create regole.")


#metodo per popolare il nuovo dataset
def create_csv(output_file):
    prolog = Prolog()
    prolog.consult("knowledge_base.pl")

    # Colonne del nuovo file CSV
    headers = [
        "City", "Country", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
        "Temperature", "Humidity", "Wind Speed", "Air_Quality", "Air_Quality_Category",
        "Year", "Month", "Day", "Monthly_Avg_Temperature", "Monthly_Avg_Wind_Speed",
        "HasRained", "Is_Stagnant", "Dispersion_Index"
    ]

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        #set per impedire la presenza di duplicati nel file
        seen_rows = set()

        # Query per ottenere i fatti
        query = """
            get_measurements(City, Month, Day, PM2_5, PM10, NO2, SO2, O3, CO, Temperature, Humidity, Wind_Speed, AQI_value, AQI_category),
            monthly_averages(City, Month, Avg_Temperature, Avg_Wind_Speed),
            geography(City, Country).
        """

        # Esecuzione della query
        results = list(prolog.query(query))

        for result in results:
            city = result['City']
            country = result['Country']
            month = result['Month']
            day = result['Day']
            pm2_5 = result['PM2_5']


            if (city, month, day,pm2_5) in seen_rows:
                continue
            else:
                seen_rows.add((city, month, day,pm2_5))


            pm10 = result['PM10']
            no2 = result['NO2']
            so2 = result['SO2']
            co = result['CO']
            o3 = result['O3']
            temperature = result['Temperature']
            humidity = result['Humidity']
            wind_speed = result['Wind_Speed']
            air_quality = result['AQI_value']
            air_quality_category = result['AQI_category']
            avg_temp = result['Avg_Temperature']
            avg_wind_speed = result['Avg_Wind_Speed']

            # Esecuzione delle regole p
            rainy_day_query = f"rainy_day('{city}', {month}, {day}, HasRained)."
            rainy_day_result = list(prolog.query(rainy_day_query))
            has_rained = rainy_day_result[0]['HasRained']

            stagnant_air_query = f"stagnant_air('{city}', {month}, {day}, Is_Stagnant)."
            stagnant_air_result = list(prolog.query(stagnant_air_query))
            is_stagnant = stagnant_air_result[0]['Is_Stagnant']

            dispersion_query = f"pollutant_dispersion('{city}', {month}, {day}, Dispersion_Index)."
            dispersion_result = list(prolog.query(dispersion_query))
            dispersion_index = round(dispersion_result[0]['Dispersion_Index'], 2)

            #Scrittura delle righe del csv
            writer.writerow([
                city, country, month, day, pm2_5, pm10, no2, so2, co, o3, temperature, humidity, wind_speed,
                air_quality, air_quality_category, avg_temp, avg_wind_speed,
                has_rained, is_stagnant, dispersion_index
            ])

    print("csv creato.")

