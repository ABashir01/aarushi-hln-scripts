import pandas as pd

data_path = 'data/treatment_outcome_lunch.csv'
df = pd.read_csv(data_path)

feed_list = ["bucketFeed", "trendingFeed", "control"]
views_map = {"total_views_pre": "Total Views (December 2020)", "total_views": "Total Views (June 2021)", "hln_views_pre": "Local News Views (December 2020)", "hln_views": "Local News Views (June 2021)"}

result_map = {}

for k,v in views_map.items():
    result_map[v] = []

    print("-------------------------------------------")
    print(v,": ")
    print("-------------------------------------------")

    for feed in feed_list:
        print("\n")
        print(feed, ": ")
        print("*********************************************")
        print(df[df["treatment"] == feed][k].describe())
        print("*********************************************")

    print("-------------------------------------------")


