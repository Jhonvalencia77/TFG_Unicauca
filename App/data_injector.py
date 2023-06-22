import os
import pandas as pd
import numpy as np

class Injector:
    BASEPATH = "injected_data/"
    dfFINAL0_5 = BASEPATH + "df_Final_Lun0_5.csv"
    dfFINAL6_11 = BASEPATH + "df_Final_Lun6_11.csv"
    dfFINAL12_17 = BASEPATH + "df_Final_Lun12_17.csv"
    dfFINAL18_23 = BASEPATH + "df_Final_Lun18_23.csv"

    def __init__(self, verbose=True):
        #Llamamos como propiedad del objeto al dataframe df_Final_Lun0_5.csv
        self.dfFinal0_5 = pd.read_csv(self.dfFINAL0_5 )
        # self.dfFinal0_5 = self.dfFinal0_5.reset_index()

        self.dfFinal6_11 = pd.read_csv(self.dfFINAL6_11 )
        # self.dfFinal6_11 = self.dfFinal6_11.reset_index()

        self.dfFinal12_17 = pd.read_csv(self.dfFINAL12_17 )
        # self.dfFinal12_17 = self.dfFinal12_17.reset_index()

        self.dfFinal18_23 = pd.read_csv(self.dfFINAL18_23 )
        # self.dfFinal18_23 = self.dfFinal18_23.reset_index()
