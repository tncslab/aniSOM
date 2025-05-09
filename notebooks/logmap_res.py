
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('./results/final/logmaps_res.csv')
means = df.groupby('method')[['r']].median()
means = means.sort_values(by='r')
print(means)
cols = means.index.tolist()




plt.figure()
sns.boxplot(data=df, x='method', y='duration', order=cols)

# rotate xticklabels
plt.xticks(rotation=90)
plt.yscale('log')

plt.ylabel('Duration (s)')
plt.xlabel('Method')
plt.grid(True)
plt.tight_layout()
plt.savefig('./notebooks/logmap_durations.png')
plt.show()

# figure about the 'r' values (same as previous)
plt.figure()
sns.boxplot(data=df, x='method', y='r', order=cols)

# rotate xticklabels
plt.xticks(rotation=90)
# plt.yscale('log')

plt.ylabel('R')
plt.xlabel('Method')
plt.grid(True)
plt.tight_layout()
plt.savefig('./notebooks/logmap_rs.png')
plt.show()