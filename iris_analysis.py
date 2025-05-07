import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# Exploration
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Grouped means
print(df.groupby('species').mean())

# Visualizations
df.iloc[:, :4].mean(axis=1).plot(title="Average Measurement Per Observation")
plt.xlabel("Sample Index")
plt.ylabel("Average Measurement")
plt.grid(True)
plt.tight_layout()
plt.savefig("line_chart.png")
plt.close()

df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar', title="Average Sepal Length per Species")
plt.ylabel("Sepal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.close()

df['petal length (cm)'].plot(kind='hist', bins=20, title="Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal vs Petal Length by Species")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()