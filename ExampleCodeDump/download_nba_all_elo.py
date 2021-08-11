import requests

download_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
target_csv_path = "nba_all_elo.csv"

response = requests.get(download_url)
response.raise_for_status()
with open(target_csv_path, "wb") as f:
    f.write(response.content)
print("Download ready.")

# runs everything in import_tests but only sampleFunc kept as function object for use in this module
#from import_tests import sampleFunc
#sampleFunc()
#
# import import_tests
# import_tests.sampleFunc()
#
# import import_tests as imtest
# imtest.sampleFunc()


