import pandas as pd

# Series is a list with both values and index (identifiers)
revenues = pd.Series([5555, 7000, 1980])

revenues.values
revenues.index

# index can be strings as wel to specific rows, similar to dictionaries
city_revenues = pd.Series(
    [4200, 8000, 6500],
    index = ["Amsterdam", "Toronto", "Tokyo"]
)
city_revenues

# making a Series using a dictionary, key -> index : values
city_employee_count = pd.Series({"Amsterdam": 5, "Tokyo": 8})

# using the keys in Series to check for presence
city_employee_count.keys()
"Tokyo" in city_employee_count
"New York" in city_employee_count

# use dictionary in DataFrame constructor, uses keys to make column names
# uses the indices of each individual series to combine into a dataframe
city_data = pd.DataFrame({
    "revenue" : city_revenues,
    "employee_count": city_employee_count
})
city_data

city_data.index
city_data.values
city_data.axes
# row indices
city_data.axes[0]
# column names
city_data.axes[1]

city_data.keys()
"Amsterdam" in city_data
"revenue" in city_data

city_revenues["Toronto"]
city_revenues[1]
city_revenues[-1]
city_revenues[1:]
city_revenues["Toronto":]

city_data.loc["Amsterdam"]
city_data.loc["Tokyo":"Toronto"]
city_data.iloc[1]

city_data.loc["Amsterdam": "Tokyo", "revenue"]

city_revenues.sum()
city_revenues.max()
