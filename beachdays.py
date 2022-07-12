# Library Imports

import json
import os
import csv
import sys
import datetime
import pandas as pd
from urllib import error, request

# File Imports

import config  # File Containing the API KEY
import predictionModel  # Training Model

# Global Variables

BASE_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# Functions


def datachecker():
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data\\usercities")
    return


def city_checker(user):


    file = (f"data\\usercities\\{user}city.txt")
    if(os.path.exists(file)):
        city = get_city(file)
    else:
        city = "None"

    while True:
        answer = input(f"Current City: {city}\nWant to change city? (Yes/No): ")
        print('')
        if(answer == "Yes"):
            city = input("City: ")
            print('\n')
            cityfile = open(file, "w")
            city = city.replace(" ", "")
            cityfile.write(city)
            cityfile.close()
            return get_city(file)

        elif(answer == "No"):
            return get_city(file)

        else:
            print('Be sure to write "Yes" or "No".')


def get_city(file):

    cityfile = open(file, "r")
    city = cityfile.read()
    cityfile.close()

    return city


def user_checker():

    if(os.path.exists("data\\user.txt")):
        user = get_user()
    else:
        user = "None"

    while True:
        answer = input(f"Current User: {user}\nWant to change user? (Yes/No): ")
        print('')
        if(answer == "Yes"):
            user = input("Username: ")
            print('\n')
            userfile = open("data\\user.txt", "w")
            user = user.replace(" ", "")
            userfile.write(user)
            userfile.close()
            return get_user()

        elif(answer == "No"):
            return get_user()

        else:
            print('Be sure to write "Yes" or "No".')


def get_user():

    userfile = open("data\\user.txt", "r")
    user = userfile.read()
    userfile.close()

    return user


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
        datetime.date.today(),
        weather_data['weather'][0]['description'],
        weather_data['dt'],
        weather_data['main']['temp'],
        weather_data['main']['pressure'],
        weather_data['main']['humidity'],
        weather_data['wind']['speed'],
        weather_data['wind']['deg'],
        beachday]

    return csv_data


def save_to_csv(weather_data, weatherdatacsv, beachday):

    # Write into the csv file the previously prepared data using the csv module, in case it exists
    # If not, we create the file, write the header and then the data

    if(not os.path.exists(weatherdatacsv)):
        file = open(weatherdatacsv, "a", newline='')
        writer = csv.writer(file)
        writer.writerow(["day", "desc", "daytime", "temperature", "pressure",
                        "humidity", "wind_str", "wind_deg", "beachday?"])
        file.close()

    file = open(weatherdatacsv, "a", newline='')
    writer = csv.writer(file)
    writer.writerow(api_to_cvs_data(weather_data, beachday))
    file.close()


def api_to_dataframe(weather_data):

    # Gotta be sure to make a dictionary with the syntax: {'column': value (inside a list)}
    df_data = {
        'day': datetime.date.today(),
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


def checkFileLines(weatherdatacsv):

    if(not os.path.exists(weatherdatacsv)):
        weathercsv = open(weatherdatacsv, "a", newline='')
        writer = csv.writer(weathercsv)
        writer.writerow(["day", "desc", "daytime", "temperature",
                        "pressure", "humidity", "wind_str", "wind_deg", "beachday?"])
        weathercsv.close()

    file = open(weatherdatacsv, "r")
    reader = csv.reader(file)
    lines = len(list(reader))
    return lines


def predictBeachDay(weather_data, weatherdatacsv):

    df_data = api_to_dataframe(weather_data)

    weatherEncoded, encoder = predictionModel.loadPreprocessData(
        weatherdatacsv)

    DecisionModel = predictionModel.modelTraining(weatherEncoded)

    df_data['desc'] = encoder.fit_transform(df_data['desc'])

    df_data = df_data.drop("day", axis=1)

    prediction = DecisionModel.predict(df_data)

    return prediction


if __name__ == "__main__":

    datachecker()
    user = user_checker()
    city = city_checker(user)
    if (not os.path.exists(f"data\{user}weatherfiles")):
        os.makedirs(f"data\{user}weatherfiles")
    weatherdatacsv = (f"data\{user}weatherfiles\{city}weatherdata.csv")

    # Get the weather data from the API

    query = weather_query(city)
    weather_data = get_weather_data(query)

    # Ask if the user wants to predict the beach day or not, and if so, predict it

    while True:

        testday = input('Want to predict if it is a beach day? (Yes/No): ')
        print('')

        if (testday == 'Yes'):

            lines = checkFileLines(weatherdatacsv)

            if lines < 11:
                print("You need at least 10 days of data to predict a beach day.")
                print('')
                break
            else:
                beachdaypred = predictBeachDay(weather_data, weatherdatacsv)

            if beachdaypred:
                print("It's a beach day! Enjoy!")
            else:
                print("Not one of the best days :c")
            break

        elif (testday == 'No'):
            break

        else:
            print('Be sure to write "Yes" or "No".')
            print('')

    # Now ask if it is a beach day or not, and if so, save it in the csv file for future predictions
    # If we answer NA, we don't save the data

    while True:

        beachday = input("Is it a beach day? (Yes/No/NA): ")
        print('')

        if (beachday == 'Yes'):
            save_to_csv(weather_data, weatherdatacsv, True)
            print("Data saved for future predictions!")
            break

        elif (beachday == 'No'):
            save_to_csv(weather_data, weatherdatacsv, False)
            print("Data saved for future predictions!")
            break

        elif (beachday == 'NA'):
            print("See you next time!")
            break

        else:
            print('Be sure to write "Yes", "No" or "NA".')
            print('')
