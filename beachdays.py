# Library Imports

import json
import os
import csv
import sys
import pandas as pd
from urllib import error, request

# File Imports

import config  # File Containing the API KEY
import predictionModel  # Training Model

# Global Variables

BASE_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# Functions


def city_checker():

    # Verifies if a city has been saved in the city.txt file or not

    if(os.path.exists("city.txt")):
        return get_city()

    # If the file doesn't exist, we ask the user to enter a city and save it in the city.txt file

    else:
        city = input("What city do you want to predict beach days?\n")
        cityfile = open("city.txt", "w")
        cityfile.write(city)
        cityfile.close()

    # Then we return whatever city is in the file

    return get_city()


def get_city():

    # Read the city from the city.txt file and return it

    cityfile = open("city.txt", "r")
    city = cityfile.read()
    cityfile.close()

    return city


def weather_query(city):

    # Build the query url to fetch the weather data

    url = (f"{BASE_API_URL}?q={city}&units=metric&appid={config.APIKEY}")

    return url


def get_weather_data(query_url):

    # Fetch the weather data from the API

    try:
        query_call = request.urlopen(query_url)
    except error.HTTPError:
        sys.exit("Can't fetch weather data.")

    data = query_call.read()

    return json.loads(data)


def api_to_cvs_data(weather_data, beachday):

    # We pick the data we want to save in the csv file, make it into a list return it

    csv_data = [
        weather_data['weather'][0]['description'],
        weather_data['dt'],
        weather_data['main']['temp'],
        weather_data['main']['pressure'],
        weather_data['main']['humidity'],
        weather_data['wind']['speed'],
        weather_data['wind']['deg'],
        beachday]

    return csv_data


def save_to_csv(weather_data, beachday):

    # Write into the csv file the previously prepared data using the csv module, in case it exists
    # If not, we create the file, write the header and then the data

    if(not os.path.exists("weatherdata.csv")):
        weathercsv = open("weatherdata.csv", "a", newline='')
        writer = csv.writer(weathercsv)
        writer.writerow(["desc", "daytime", "temperature", "pressure",
                        "humidity", "wind_str", "wind_deg", "beachday?"])
        weathercsv.close()

    weathercsv = open("weatherdata.csv", "a", newline='')
    writer = csv.writer(weathercsv)
    writer.writerow(api_to_cvs_data(weather_data, beachday))
    weathercsv.close()


def api_to_dataframe(weather_data):

    # Gotta be sure to make a dictionary with the syntax: {'column': value (inside a list)}

    df_data = {
        'desc': [weather_data['weather'][0]['description']],
        'daytime': [weather_data['dt']],
        'temperature': [weather_data['main']['temp']],
        'pressure': [weather_data['main']['pressure']],
        'humidity': [weather_data['main']['humidity']],
        'wind_str': [weather_data['wind']['speed']],
        'wind_deg': [weather_data['wind']['deg']]
    }

    # So when converting the dictionary into a dataframe, it will be in the desired format

    df_data = pd.DataFrame(df_data)

    return df_data


def checkFileLines():

    if(not os.path.exists("weatherdata.csv")):
        weathercsv = open("weatherdata.csv", "a", newline='')
        writer = csv.writer(weathercsv)
        writer.writerow(["desc", "daytime", "temperature",
                        "pressure", "humidity", "wind_str", "wind_deg", "beachday?"])
        weathercsv.close()

    file = open("weatherdata.csv", "r")
    reader = csv.reader(file)
    lines = len(list(reader))
    return lines


def predictBeachDay(weather_data):

    df_data = api_to_dataframe(weather_data)

    weatherEncoded, encoder = predictionModel.loadPreprocessData()

    DecisionModel = predictionModel.modelTraining(weatherEncoded)

    df_data['desc'] = encoder.fit_transform(df_data['desc'])

    prediction = DecisionModel.predict(df_data)

    return prediction


if __name__ == "__main__":

    # Get the city from the city.txt file or ask the user to enter a city

    city = city_checker()

    # Get the weather data from the API

    query = weather_query(city)
    weather_data = get_weather_data(query)

    # Ask if the user wants to predict the beach day or not, and if so, predict it

    while True:

        testday = input('Want to predict if it is a beach day? (Yes or No): ')

        if (testday == 'Yes'):

            lines = checkFileLines()

            if lines < 11:
                print("You need at least 10 days of data to predict a beach day.")
                break
            else:
                beachdaypred = predictBeachDay(weather_data)

            if beachdaypred:
                print("It's a beach day! Enjoy!")
            else:
                print("Not one of the best days :c")
            break

        elif (testday == 'No'):
            break

        else:
            print('Be sure to write "Yes" or "No".')

    # Now ask if it is a beach day or not, and if so, save it in the csv file for future predictions
    # If we answer NA, we don't save the data

    while True:

        beachday = input("Is it a beach day? (Yes, No or NA): ")

        if (beachday == 'Yes'):
            save_to_csv(weather_data, True)
            print("Data saved for future predictions!")
            break

        elif (beachday == 'No'):
            save_to_csv(weather_data, False)
            print("Data saved for future predictions!")
            break

        elif (beachday == 'NA'):
            print("See you next time!")
            break

        else:
            print('Be sure to write "Yes", "No" or "NA".')
