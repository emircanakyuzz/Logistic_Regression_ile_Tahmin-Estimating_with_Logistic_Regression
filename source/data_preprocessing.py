import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class DataProcessing:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)

    def add_output_column(self):
        self.df["output"] = 0

    def output_values(self, target_column, output_column, output_value):
        if output_column not in self.df.columns:
            self.df[output_column] = None
        
        for index, value in enumerate(self.df[target_column]):
            if pd.notna(value):
                self.df.loc[index, output_column] = output_value
        return self.df

    def convert_categorical_to_numeric(self):
        self.df["payment_type"] = self.df.payment_type.map({'H': 0, 'S': 1})
        self.df["transaction_type"] = self.df.transaction_type.map({"NEFT": 0, "NMSC": 1, "NTRF": 2, "NCHG": 3, "NCHK": 4, "NTDP": 5, "NVRM": 6, "NTAX": 7})

    def handle_outliers(self, outlier_columns_list):
        for column in outlier_columns_list:
            Q1 = np.percentile(self.df.loc[:, column], 25)
            Q3 = np.percentile(self.df.loc[:, column], 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            upper = np.where(self.df.loc[:, column] >= upper_bound)[0]
            lower = np.where(self.df.loc[:, column] <= lower_bound)[0]
            
            median = self.df.loc[:, column].median()
            self.df.loc[upper, column] = median
            self.df.loc[lower, column] = median

    def split_features_and_target(self):
        X = self.df[["company_code", "payment_type", "amount", "transaction_type"]]
        y = self.df[["output"]]
        return X, y

    def scale_features(self, X):
        scaler = StandardScaler()
        X["amount"] = scaler.fit_transform(X[["amount"]])
        return X

    def get_features_and_target(self):
        self.add_output_column()
        self.output_values("customer_number", "output", 1)
        self.output_values("main_account", "output", 2)
        self.convert_categorical_to_numeric()
        self.handle_outliers(["amount", "document_number"])
        X, y = self.split_features_and_target()
        X = self.scale_features(X)
        return X, y

# Veri Ön İşleme adımları
data_processor = DataProcessing("data.xlsx")
X, y = data_processor.get_features_and_target()

print("Veri Ön İşleme adımları başarılı bir şekilde tamamlandı.")
print(X.head())
print(y.head())