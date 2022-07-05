
# Weather API Specifications

# It takes a city and an optional flag to display the output in celsius or fahrenheit as input;
# We call a weather API to fetch the specified city weather data;
# We save the fetched data into a CSV file for model training;

import json
import os
import csv
import sys
from urllib import error, parse, request
from pprint import pp

import config  # File Containing the API KEY
import predictionModel  # Training Model

BASE_API_URL = "http://api.openweathermap.org/data/2.5/weather"


def city_checker():
    if(os.path.exists("city.txt")):
        return get_city()

    else:
        city = input("What city do you want to predict beach days?\n")
        cityfile = open("city.txt", "w")
        cityfile.write(city)
        cityfile.close()

    return get_city()


def get_city():

    cityfile = open("city.txt", "r")
    city = cityfile.read()
    cityfile.close()

    return city


def weather_query(city):

    url = (f"{BASE_API_URL}?q={city}&units=metric&appid={config.APIKEY}")

    return url


def get_weather_data(query_url):

    try:
        query_call = request.urlopen(query_url)
    except error.HTTPError:
        sys.exit("Can't fetch weather data.")

    data = query_call.read()

    return json.loads(data)


def api_to_cvs_data(weather_data, beachday):

    csv_data = [
        weather_data['weather'][0]['description'],
        weather_data['main']['temp'],
        weather_data['main']['pressure'],
        weather_data['main']['humidity'],
        weather_data['wind']['speed'],
        weather_data['wind']['deg'],
        beachday]

    return csv_data


def save_to_csv(weather_data, beachday):

    weathercsv = open("weatherdata.csv", "a", newline='')
    writer = csv.writer(weathercsv)
    writer.writerow(api_to_cvs_data(weather_data, beachday))
    weathercsv.close()

def predictBeachDay(weather_data):

    # Function to load and preprocess dataset

    weatherEncoded = predictionModel.loadPreprocessData()

    # Function that calls and fits the prediction model

    DecisionModel = predictionModel.modelTraining(weatherEncoded)

    # Function that predicts with the weather_data of today

    prediction = DecisionModel.predict(weather_data) # Work in progress

    return prediction # Boolean


if __name__ == "__main__":

    while True:
        beachday = input("Is it a beach day? (True of False): ")
        if (beachday == 'True' or beachday == 'False'):
            break
        else:
            print('Be sure to write "True" or "False".')

    city = city_checker()

    query = weather_query(city)
    weather_data = get_weather_data(query)

    save_to_csv(weather_data, beachday)

    while True:

        testday = input('Want to predict if it is a beach day? (Yes or No): ')

        if (testday == 'Yes'):

            beachdaypred = predictBeachDay(api_to_cvs_data(weather_data, beachday))

            if beachdaypred: 
                print("It's a beach day! Enjoy!") 
            else: 
                print("Not one of the best days :c")
            break

        elif (testday == 'No'):
            break

        else:
            print('Be sure to write "Yes" or "No".')
