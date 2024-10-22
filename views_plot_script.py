import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mstats
import starbars

data_path = 'data/treatment_outcome_lunch.csv'
df = pd.read_csv(data_path)

# # Filter the values 
df = df[df['total_views_pre'] >= 200] # Filter out total views of less than 600
# df = df[df['total_views_pre'] <= 15000]
df = df[df['avg_posts'] >= 600]
# df = df[df['avg_posts'] <= 40000]


bucket_group = df[df['treatment'] == 'bucketFeed']
trending_group = df[df['treatment'] == 'trendingFeed']
control_group = df[df['treatment'] == 'control']

bucket_group = bucket_group[bucket_group['hln_views'] <= 40]
trending_group = trending_group[trending_group['hln_views'] <= 40]

bucket_group = bucket_group[bucket_group['hln_views'] >= 1]
trending_group = trending_group[trending_group['hln_views'] >= 1]


def get_mean_and_std_err(group, name):
    mean = group[name].mean()
    std_err = group[name].std() / (len(group) ** 0.5)
    return mean, std_err

def perform_ttest(group1, group2, name):
    t_stat, p_value = ttest_ind(group1[name], group2[name], equal_var=False)
    return p_value

def make_graph(name):    
    control_mean, control_std = get_mean_and_std_err(control_group, name)
    trending_mean, trending_std = get_mean_and_std_err(trending_group, name)
    bucket_mean, bucket_std = get_mean_and_std_err(bucket_group, name)

    labels = ['Control', 'Trending', 'Bucket']
    means = [control_mean, trending_mean, bucket_mean]
    errors = [control_std, trending_std, bucket_std]
    print(control_mean)

    title = name
    if name == "hln_views":
        title = "Local Views"
    elif name == "total_views":
        title = "Total Views"

    # Perform t-tests
    p_value_trending = perform_ttest(control_group, trending_group, name)
    p_value_bucket = perform_ttest(control_group, bucket_group, name)

    fig, ax = plt.subplots(figsize=(20, 12))
    bars = ax.bar(labels, means, yerr=errors, capsize=10, color=['#5cd1dd', '#b494d0', '#f78154'], edgecolor='black')
    ax.set_ylabel(f'Average {title}', fontsize=16)
    ax.set_title(f'Average {title} with Standard Error for Control and Treatment Groups', fontsize=16)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=14)

    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=14, fontweight='bold')

    annotations = [("Control", "Trending", f'p = {p_value_trending:.4f}'), ("Control", "Bucket", f'p = {p_value_bucket:.4f}')]
    starbars.draw_annotation(annotations, fontsize=14)


    ax.grid(True, which='both', linewidth=0.5)

    plt.savefig(f"tables/final_{name}_figure.png")
    plt.show()

make_graph("hln_views")
make_graph("total_views")

all_groups = {"Control": control_group, "Bucket": bucket_group, "Trending": trending_group}
views_groups = {"Total Views (March 2021)": 'total_views', "Total Views (December 2020)": 'total_views_pre', "Local News Views (March 2021)": 'hln_views', "Local News Views (December 2020": 'hln_views_pre', "Average posts supplied in March 2021": 'avg_posts'}
for k,v in all_groups.items():
    print(f"{k}")
    print("--------------------------------------------------------------")
    for key, val in views_groups.items():
        mean = v[val].mean()
        print(f"{key}: {mean}")

    size = v.size
    print(f"User number: {size}")
    print("--------------------------------------------------------------")

total_sample_size = df.size
print(f'Total sample size is {total_sample_size}')
