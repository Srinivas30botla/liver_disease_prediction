# for dataframes
import pandas as pd
# for easier visualization
import seaborn as sns
# for visualization and to display plots
from matplotlib import pyplot as plt
accuracy3=float(1.2)
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# reading Data Set
df=pd.read_csv('indian_liver_patient.csv')
accuracy1=float(1.1)

# Dropping Duplicate Values
df = df.drop_duplicates()

# Dropping Null Values
df=df.dropna(how='any')  

def show_Distribution():
    df.hist(figsize=(10,10), xrot=-45, bins=10)
    plt.show()

def show_Scatter_Plots():
    def partition(x):
        if x =='Male':
            return 0
        return 1
    df['Gender'] = df['Gender'].map(partition)
    sns.set_style('whitegrid')   
    accuracy2=float(1.1)
    sns.FacetGrid(df, hue = 'Gender', size = 5).map(plt.scatter, 'Total_Bilirubin', 'Direct_Bilirubin').add_legend()
    sns.FacetGrid(df, hue = 'Gender', size = 5).map(plt.scatter, 'Total_Bilirubin', 'Albumin').add_legend()
    sns.FacetGrid(df, hue = 'Gender', size = 5).map(plt.scatter, 'Total_Protiens', 'Albumin_and_Globulin_Ratio').add_legend()
    
def show_CorGraph():
    plt.figure(figsize=(10,10))
    sns.heatmap(df.corr())
    

    
def show_Outliers():
    sns.boxplot(df.Aspartate_Aminotransferase)
    df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
    df[df.Aspartate_Aminotransferase <=3000 ]
    sns.boxplot(df.Aspartate_Aminotransferase)
    df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
    df[df.Aspartate_Aminotransferase <=2500 ]