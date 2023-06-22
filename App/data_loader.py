import os
import pandas as pd
import numpy as np
import ray

class TripsLoader:
    BASEPATH = "data/"
    ORIGIN_FILENAME = "timeseries_o.csv"
    DEST_FILENAME = "timeseries_d.csv"
    TRAYEC_FILENAME = BASEPATH + "Trayectos_Periodo_Confinamiento2.csv"

    def __init__(self, verbose=True):
        self.routesLoader = RoutesLoader() # Instanciamos objeto de la clase RoutesLoader()

        #Llamamos como propiedad del objeto al dataframe timeseries_o.csv
        self.timeseries_o = pd.read_csv(self.BASEPATH + self.ORIGIN_FILENAME)
        self.timeseries_o["ds"] = pd.to_datetime(self.timeseries_o["ds"])
        self.timeseries_o = self.timeseries_o.set_index("ds")
        self.timeseries_o = self.timeseries_o.astype(np.float32)

        #Llamamos como propiedad del objeto al dataframe timeseries_d.csv
        self.timeseries_d = pd.read_csv(self.BASEPATH + self.DEST_FILENAME)
        self.timeseries_d["ds"] = pd.to_datetime(self.timeseries_d["ds"])
        self.timeseries_d = self.timeseries_d.set_index("ds")
        self.timeseries_d = self.timeseries_d.astype(np.float32)

        #Llamamos como propiedad del objeto al dataframe Trayectos_Periodo_Confinamiento2
        self.trayectos = TripsLoader.load_trayectos.remote(TripsLoader.TRAYEC_FILENAME)

    @staticmethod
    @ray.remote
    def load_trayectos(path):
        datos = pd.read_csv(path)
        datos["ds"] = pd.to_datetime(datos["ds"])
        datos = datos.set_index("ds")
        return datos


class PassengersDataLoader:
    BASEPATH = "data/"
    RENFE_MONTHLY_DATA_PATH = BASEPATH + "renfe_monthly_data.csv"
    RENFE_USERS_BYSTOP = BASEPATH + "up_down_bystop.csv"
    METRO_DAILY_DATA = BASEPATH + 'metro_daily_data.csv'
    METRO_USERS_BYSTOP = BASEPATH + 'metro_up_down_bystop.csv'

    def __init__(self):
        # Llamamos como propiedad del objeto al dataframe renfe_monthly_data.csv y metro_daily_data.csv
        self.renfe_monthly_data = PassengersDataLoader.load_renfe_monthly_data(PassengersDataLoader.RENFE_MONTHLY_DATA_PATH)
        self.metro_daily_data = PassengersDataLoader.load_metro_daily_data(PassengersDataLoader.METRO_DAILY_DATA)

        # Llamamos como propiedad del objeto al dataframe up_down_bystop.csv y metro_up_down_bystop.csv
        self.renfe_users_bystop = PassengersDataLoader.load_renfe_users_bystop(PassengersDataLoader.RENFE_USERS_BYSTOP)
        self.metro_users_bystop = PassengersDataLoader.load_metro_users_bystop(PassengersDataLoader.METRO_USERS_BYSTOP)

    def update_renfe_timeseries(self,timeseries):
        self.renfe_monthly_data = self.renfe_monthly_data.append(timeseries)
        self.renfe_monthly_data = self.renfe_monthly_data[~self.renfe_monthly_data.index.duplicated(keep='last')].sort_index()


    def update_metro_timeseries(self,timeseries):
        self.metro_daily_data = self.metro_daily_data.append(timeseries)
        self.metro_daily_data = self.metro_daily_data[~self.metro_daily_data.index.duplicated(keep='last')].sort_index()

    @staticmethod
    # @ray.remote
    def load_renfe_monthly_data(path,start_month = "20130101"):
        data = pd.read_csv(path, names = ["users"])
        period = pd.period_range(start_month, periods = len(data), freq = "M")
        data["ds"] = period
        data = data.set_index("ds")
        return data

    @staticmethod
    # @ray.remote
    def load_metro_daily_data(path):
        return pd.read_csv(path,
                           delimiter = ';',
                           converters = {
                               'ds': lambda x: pd.to_datetime(x, dayfirst=True)
                           },
                           index_col = 'ds')

    @staticmethod
    # @ray.remote
    def load_renfe_users_bystop(path):
        return pd.read_csv(path)

    @staticmethod
    # @ray.remote
    def load_metro_users_bystop(path):
        return  pd.read_csv(path)


class RoutesLoader:
    BASEPATH = "data/"
    ROUTES_FILENAME = "routes.csv"

    def __init__(self):
        # Llamamos como propiedad del objeto al dataframe routes.csv
        self.routes = pd.read_csv(RoutesLoader.BASEPATH
                                  + RoutesLoader.ROUTES_FILENAME,
                                  dtype={"stop_district":str})

        # Propiedad del objeto que contiene una lista de los distritos donde hay una parada de cercanías o de metro
        self.districts = self.routes.stop_district.unique()

        # Propiedad del objeto que contiene la lista ['cercanias' 'metro' 'emt'] (dtype=objeto)
        self.services = self.routes.service.unique()

        # Contiene un diccionario con las lineas de cada servicio {'cercanias': array(['C1', 'C10', 'C2', 'C3', 'C4', 'C5', 'C8', 'C9']),
        self.lines = {}
        for service in self.services:
            self.lines[service] = self.routes[self.routes.service == service].line.unique()
