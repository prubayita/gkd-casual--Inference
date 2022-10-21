import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, LabelEncoder
# Importing the SimpleImputer class from sklearn
from sklearn.impute import SimpleImputer

class Cleaner():
    '''
    This class contains helper functions to explore data and functions to clean data in a pandas dataframe.
    '''
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_numerical_columns(df: pd.DataFrame) -> list:
        '''
        Function to output numerical columns 
        Args:
        pandas Dataframe
        '''
        numerical_columns = df.select_dtypes(include='number').columns.tolist()
        return numerical_columns

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> list:
        '''
        Function to output categorical columns 
        Args:
        pandas Dataframe
        '''
        categorical_columns = df.select_dtypes(
            include=['object']).columns.tolist()
        return categorical_columns

    def percent_missing(self, df):
        """
        Print out the percentage of missing entries in a dataframe
        Args: 
        """
        # Calculate total number of cells in dataframe
        totalCells = np.product(df.shape)

        # Count number of missing values per column
        missingCount = df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("The dataset contains", round(
            ((totalMissing/totalCells) * 100), 2), "%", "missing values.")
    
    def drop_columns(self, columns):
        '''
        Function that drops columns 
        Args:
        pandas dataframe columns
        '''

        self.df.drop(columns, axis=1, inplace=True)

    def convert_to(self, df,columns, data_type):
        '''
        Convert Columns to desired data types.
        '''

        for column in columns:
            df[column] = self.df[column].astype(data_type)
        
        return df
    

    def fill_numerical_column(self, column):
        '''
        Function to fill Numerical null values with 
        mean or median depending on the skewness of the column
        '''

        skewness = self.df[column].skew()
        if((-1 < skewness) and (skewness < -0.5)):
            # Negative skew
            self.df[column].fillna(self.df[column].mean())

        elif((0.5 < skewness) and (skewness < 1)):
            # Positive skew
            self.df[column].fillna(self.df[column].median())

        else:
            # highly skewed 
            self.df[column].fillna(self.df[column].median())
    
    def fill_missing_median(df):
        """
        # Function to fill categorical nulls with median
        """
        col=['CompetitionDistance','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
        for column in col:
            imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
            imp_median = imp_median.fit(df[[column]])
            df[column] = imp_median.transform(df[[column]]).ravel()
    
        return df
    
    def fill_missing_zero(df):
        """
        # Function to fill categorical nulls with 0
        """
        col=['Promo2SinceWeek','Promo2SinceYear']
        for column in col:
            imp_median = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
            imp_median = imp_median.fit(df[[column]])
            df[column] = imp_median.transform(df[[column]]).ravel()
        
        return df
    def fill_missing_mode(df):
        """
        # Function to fill categorical nulls with mode
        """
        col=['PromoInterval']
        for column in col:
            imp_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imp_median = imp_median.fit(df[[column]])
            df[column] = imp_median.transform(df[[column]]).ravel()
    
        return df

    def remove_null_row(self, df: pd.DataFrame, columns: str) -> pd.DataFrame:
        for column in columns:
            df = df[~ df[column].isna()]

        return df


    def fix_outliers(self, col):
        '''
        Handle outliers of specified column
        '''
        q1 = self.df[col].quantile(0.25)
        q3 = self.df[col].quantile(0.75)

        lower_bound = q1 - ((1.5) * (q3 - q1))
        upper_bound = q3 + ((1.5) * (q3 - q1))

        self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
        self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
    
    def normal_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        scaller = StandardScaler()
        scalled = pd.DataFrame(scaller.fit_transform(
            df[self.get_numerical_columns(df)]))
        scalled.columns = self.get_numerical_columns(df)

        return scalled

    def minmax_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        scaller = MinMaxScaler()
        scalled = pd.DataFrame(
            scaller.fit_transform(
                df[self.get_numerical_columns(df)]),
            columns=self.get_numerical_columns(df)
        )

        return scalled

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        normalizer = Normalizer()
        normalized = pd.DataFrame(
            normalizer.fit_transform(
                df[self.get_numerical_columns(df)]),
            columns=self.get_numerical_columns(df)
        )

        return normalized
    def encoding_data(self, df):
        for column in df.columns:
            if df[column].dtype == np.int64 or df[column].dtype == np.float64:
                continue
            df[column] = LabelEncoder().fit_transform(df[column])
        
        return df


    def save_clean(self,df):
        try:
            df.to_csv('../data/clean_breast_cancer.csv', index=False)
        except:
            print('Log: Error while Saving File')
    
    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This runs a series of cleaner methods on the df passed to it. 
        """
        
        
        return df