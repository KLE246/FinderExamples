import matplotlib as matplotlib
import pandas as pd
from pandas_ds import *

further_city_data = pd.DataFrame(
    {"revenue": [7000, 3400], "employee_count":[2,2]},
    index = ["New York", "Barcelona"]
)

#default for concat is join along axis = 0, appends rows like rbind()
all_city_data = pd.concat([city_data, further_city_data], sort = False)

city_countries = pd.DataFrame({
    "country":  ["Holland", "Japan", "Holland", "Canada", "Spain"],
    "capital": [1, 1, 0, 0, 0]},
    index = ["Amsterdam", "Tokyo", "Rotterdam", "Toronto", "Barcelona"]
)

cities = pd.concat([all_city_data, city_countries], axis = 1, sort = False)
cities

# use inner join so that only common indices are in the final dataset
pd.concat([all_city_data, city_countries], axis = 1, join = "inner")

countries = pd.DataFrame({
    "population_millions": [17,127,37],
    "continent": ["Europe", "Asia", "North America"]
}, index = ["Holland", "Japan", "Canada"])

# left_on used to join on "country", all dissimilar values in country are not included, an inner join is done by default
pd.merge(cities, countries, left_on = "country", right_index = True)

# how = "left" for left join
pd.merge(cities, countries, left_on = "country", right_index = True, how = "left")

