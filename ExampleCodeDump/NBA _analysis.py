import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 1000)
pd.set_option("display.precision", 2)

nba = pd.read_csv("nba_all_elo.csv")
print(type(nba))
print(len(nba))
print(nba.shape)
print(nba.head())
print(nba.info())
print(nba.describe())

print(nba.describe(include=object))

print(nba["team_id"].value_counts())
print(nba["fran_id"].value_counts())

# similar to filter but doesn't change the dataset
# create boolean array with all "Lakers" fran_id being true
#  data.frame.loc[[rows indices or boolean array], [column names]]
nba.loc[nba["fran_id"] == "Lakers", "team_id"].value_counts()

nba["date_played"] = pd.to_datetime(nba["date_game"])
nba.loc[nba["team_id"] == "MNL", "date_played"].min()

nba.loc[nba['team_id'] == "MNL", "date_played"].max()
nba.loc[nba["team_id"] == "MNL", "date_played"].agg(("min", "max"))

nba.loc[nba["team_id"] == "BOS", "pts"].sum()

"points" in nba.keys()
"pts" in nba

nba.iloc[-2]
nba.loc[5555:5559, ["fran_id", "opp_fran", "pts", "opp_pts"]]


# Querying dataset
# similar to filters
current_decade = nba[nba["year_id"] >2010]
current_decade.shape

games_with_notes = nba[nba["notes"].notnull()]
games_with_notes.shape

ers = nba[nba["fran_id"].str.endswith("ers")]
ers.shape

nba[
    (nba["_iscopy"] == 0) &
    (nba["pts"] > 100) &
    (nba["opp_pts"] > 100) &
    (nba["team_id"] == "BLB")
]

nba.info()
nba["game_location"]
nba[
    (nba["_iscopy"] == 0) &
    (nba["team_id"].str.startswith("LA")) &
    (nba["date_game"].str.endswith("1992")) &
    (nba["notes"].notnull())
]


# similar to group by and summarize
nba.groupby("fran_id", sort = False)["pts"].sum()
nba[
    (nba["fran_id"] == "Spurs") &
    (nba["year_id"] > 2010)
].groupby(["year_id", "game_result"])["game_id"].count()

nba[
    (nba["fran_id"] == "Warriors") &
    (nba["year_id"] == 2015)
].groupby(["is_playoffs", "game_result"])["game_id"].count()

df = nba.copy()
df.shape
# similar to mutate
df["difference"] = df.pts - df.opp_pts
df.shape

df["difference"].max()

renamed_df = df.rename(
    columns = {"game_result": "result", "game_location": "location"}
)

renamed_df.info()
df.shape
elo_columns = ["elo_i", "elo_n", "opp_elo_i", "opp_elo_n"]
df.drop(elo_columns, inplace = True, axis =1)
df.shape

df["date_game"] = pd.to_datetime(df["date_game"])

df["game_location"].nunique()
df["game_location"].value_counts()
df["game_location"] = pd.Categorical(df["game_location"])
df["game_location"].dtype

df.info()

df["lg_id"].value_counts()
df["lg_id"] = pd.Categorical(df["lg_id"])
df["game_result"] = pd.Categorical(df["game_result"])

rows_without_missing_data = nba.dropna()
rows_without_missing_data.shape

data_without_missing_columns = nba.dropna(axis =1)
data_without_missing_columns.shape

data_with_default_notes = nba.copy()
data_with_default_notes["notes"].fillna(
    value = "no notes at all",
    inplace = True
)
data_with_default_notes.describe()
nba.shape
nba.tail(3)
nba.head()

# example of one method for dropping rows that fit criteria
toRemove = nba.index[nba["pts"] == 0].tolist()
nba.drop(toRemove, inplace = True)
nba.describe()

# example checks for consistency
nba[(nba["pts"] > nba["opp_pts"]) & (nba["game_result"] == "L")].empty
nba[(nba["pts"] < nba["opp_pts"]) & (nba["game_result"] == "W")].empty

nba[nba["fran_id"] == "Knicks"].groupby("year_id")["pts"].sum().plot()
nba["fran_id"].value_counts().head(10).plot(kind = "bar")

nba[
    (nba["fran_id"] == "Heat") &
    (nba["year_id"] == 2013)
]["game_result"].value_counts().plot(kind = "pie")


