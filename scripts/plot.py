import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plot():
    def __init__(self) -> None:
        pass

    def plot_bar(self, df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str, ax)->None:
        plt.figure(figsize=(12, 7))
        sns.barplot(data = df, x=x_col, y=y_col, ax=ax)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks( fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        return plt.show()

    def plot_heatmap(self, df:pd.DataFrame, title:str, cbar=False)->None:
        plt.figure(figsize=(12, 7))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
        plt.title(title, size=18, fontweight='bold')
        return plt.show()


    def plot_pie(self, data, column, title:str):
        plt.figure(figsize=(12, 7))
        count = data[column].value_counts()
        colors = sns.color_palette('pastel')[0:5]
        plt.pie(count, labels = count.index, colors = colors, autopct='%.0f%%')
        plt.title(title, size=18, fontweight='bold')
        return plt.show()


    def plot_hist(self, df:pd.DataFrame, column:str, color:str)->None:
        plt.figure(figsize=(9, 7))
        sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        return plt.show()

    def bivariate_plot(self,df,features, fields):
        fig, axs = plt.subplots(10,3, figsize=(20,45))
        for col in range(len(features)):  
            for f in range(len(fields)):  
                sns.histplot(df, 
                            x=features[col]+"_"+fields[f], 
                            hue="diagnosis", element="bars", 
                            stat="count", 
                            palette=["gold", "purple"],
                            ax=axs[col][f])
        