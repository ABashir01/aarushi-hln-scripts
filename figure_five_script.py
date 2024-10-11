import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/treatment_outcome_lunch.csv')

print("Reading CSV")

# Create individual KDE plots with common_norm=False for separate normalization
sns.kdeplot(data=(df[df["treatment"] == "trendingFeed"].hln_views), label='trendingFeed', fill=True, common_norm=False, bw_adjust=3)
sns.kdeplot(data=(df[df["treatment"] == "bucketFeed"].hln_views), label='bucketFeed', fill=True, common_norm=False,  bw_adjust=3)
sns.kdeplot(data=(df[df["treatment"] == "control"].hln_views), label='control', fill=True, common_norm=False, bw_adjust=1)

print("Plot Made")

# Labels and title
plt.title('Views on Hyperlocal News')
plt.xlabel('Views on Hyperlocal News')
plt.ylabel('Density')

# Set ticks and limits
plt.yticks([i * 0.05 for i in range(6)])
plt.xticks([i * 20 for i in range(7)])
plt.ylim(0, 0.25)
plt.xlim(0, 100)

# Show legend
plt.legend(title='Treatment')

# Save the figure
# plt.savefig("final_results/figure_five.png")

# Show the plot
plt.show()
