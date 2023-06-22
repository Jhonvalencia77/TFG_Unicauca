import ray
import asyncio
import os
from data_loader import * #Importamos todas las clases del archivo data_loader
@ray.remote
class RouteTrip:
    __COEF_BASENAME = PassengersDataLoader.BASEPATH + "%s_coef.csv" #data/pt_data/%s_coef.csv

    def __init__(self,tripsloader,ptdata):
        #propiedad del objeto que recibe el parámetro tripsloader
        self.tripsLoader = tripsloader
        # Propiedad para elegir el archivo con la clase PassengersDataLoader
        self.ptdata = ptdata
        #Propiedad que contiene el archivo routes.csv
        self.routes = tripsloader.routesLoader.routes.set_index("stop_id")

    def get_users_renfe(self, timeseries_o,max_precision = True):
        #  Filtramos las filas del archivo routes.csv mediante el "stop_id" del archivo up_down_bystop.csv
        renfe_stops = self.routes.loc[self.ptdata.renfe_users_bystop[self.ptdata.renfe_users_bystop.stop_id.notna()].stop_id]
        # Se eliminan las filas con indice duplicado
        renfe_stops = renfe_stops[~renfe_stops.index.duplicated(keep='first')]

        # seleccionamos un subconjunto de columnas de timeseries_o donde están los distritos con paradas de renfe
        subtimeseries_o = timeseries_o[renfe_stops.stop_district.unique()]
        # Se filtran las filas que se encuentren dentro del rango
        # subtimeseries_o = subtimeseries_o.between_time("6:00", "0:00")

        # Td es el dataframe renfe_monthly_data.csv modificado en "ds" (Usuarios reales)
        Td = self.ptdata.renfe_monthly_data

        # Calculamos el número de usuarios por estación
        # up = self.compute_users_method2(renfe_stops,subtimeseries_o,self.ptdata.renfe_users_bystop,Td,'cercanias')
        up = self.compute_users_method1.remote(self=self,stops=renfe_stops,subtimeseries_o=subtimeseries_o,Td=Td,service='cercanias')
        if max_precision:
            up2 = self.compute_users_method2.remote(self=self,stops=renfe_stops,subtimeseries_o=subtimeseries_o,
                                                        usersbystop=self.ptdata.renfe_users_bystop,Td=Td,service='cercanias')
            up = ray.get(up)
            up2 = ray.get(up2)
            up = (up + up2)/2

        return up

    def get_users_metro(self,timeseries_o, max_precision = True):
        metro_stops = self.routes.loc[self.ptdata.metro_users_bystop[self.ptdata.metro_users_bystop.stop_id.notna()].stop_id]
        metro_stops = metro_stops[~metro_stops.index.duplicated(keep='first')]

        subtimeseries_o = timeseries_o[metro_stops.stop_district.unique()]
        # subtimeseries_o = subtimeseries_o.between_time("6:00", "0:00")

        Td = self.ptdata.metro_daily_data

        up = self.compute_users_method1.remote(self=self,stops=metro_stops,subtimeseries_o=subtimeseries_o,Td=Td,service='metro')
        if max_precision:
            up2 = self.compute_users_method2.remote(self=self,stops=metro_stops,subtimeseries_o=subtimeseries_o,
                                                    usersbystop=self.ptdata.metro_users_bystop,Td=Td,service='metro')
            up = ray.get(up)
            up2 = ray.get(up2)
            up = (up + up2) / 2

        return up
    @ray.remote
    def compute_users_method1(self,stops,subtimeseries_o,Td,service):
        # seleccion = (stops['stop_district'] == subtimeseries_o.columns[0])
        # stop = stops[seleccion]
        if service == 'cercanias':
            coef = self.compute_p(self.tripsLoader,self.ptdata.renfe_users_bystop,service)
        elif service == 'metro':
            coef = self.compute_p(self.tripsLoader,self.ptdata.metro_users_bystop,service)

        up = pd.DataFrame({}, index=subtimeseries_o.index)

        for stop_id in stops.index:
            up[stop_id] = subtimeseries_o[stops.loc[stop_id].stop_district] * coef.loc[stop_id].p_o
        p = []
        if service == 'cercanias':
            for time_period in Td.index:
                up_p = up[(up.index.month == time_period.month) &
                                                       (up.index.year == time_period.year)]


                p_co = up_p.sum().sum()/(Td.loc[time_period].iloc[0]*1e6)
                p.append(p_co)

                up[(up.index.month == time_period.month) & (up.index.year == time_period.year)] = up[(up.index.month == time_period.month) & (up.index.year == time_period.year)]/p_co


        elif service == 'metro':
            for time_period in Td.index:
                up_p = up[up.index.date == time_period.date()]

                p_co = up_p.sum().sum() / (Td.loc[time_period].iloc[0])
                p.append(p_co)

                up[up.index.date == time_period.date()] = up[up.index.date == time_period.date()] / p_co

        return up
    @ray.remote
    def compute_users_method2(self,stops,subtimeseries_o,usersbystop,Td,service):
        # seleccion = (stops['stop_district'] == subtimeseries_o.columns[0])
        # stop = stops[seleccion] #Info de la estación(es) a calcular
        # userstop = usersbystop[usersbystop['stop_id'].isin(stop.index)] #Info de up y down de la estación(es)

        beta = usersbystop
        stops = stops["stop_district"]

        beta = beta[beta.stop_id.notna()]
        beta.up = beta.up / beta.up.sum()

        up = pd.DataFrame({})
        if service == "cercanias":
            for time_period in Td.index:
                subtimeseries_op = subtimeseries_o[(subtimeseries_o.index.month == time_period.month) &
                                                   (subtimeseries_o.index.year == time_period.year)]

                # subtimeseries_o[subtimeseries_o.index.weekday >= 5] = subtimeseries_o[subtimeseries_o.index.weekday >= 5] * 0
                alpha_o = subtimeseries_op / subtimeseries_op.sum()

                # subtimeseries_d[subtimeseries_d.index.weekday >= 5] = subtimeseries_d[subtimeseries_d.index.weekday >= 5] * 0

                up_aux = pd.DataFrame({}, index=subtimeseries_op.index)

                for stop_id in beta.stop_id:

                    up_aux[stop_id] = alpha_o[stops.loc[stop_id]] * Td.loc[time_period].iloc[0] * 1e6 * \
                                      beta[beta.stop_id == stop_id].up.iloc[0]

                up = up.append(up_aux)

        elif service == 'metro':
            for time_period in Td.index:
                subtimeseries_op = subtimeseries_o[subtimeseries_o.index.date == time_period.date()]

                alpha_o = subtimeseries_op / subtimeseries_op.sum()

                up_aux = pd.DataFrame({}, index=subtimeseries_op.index)

                for stop_id in beta.stop_id:
                    up_aux[stop_id] = alpha_o[stops.loc[stop_id]] * Td.loc[time_period].iloc[0] * \
                                      beta[beta.stop_id == stop_id].up.iloc[0]

                up = up.append(up_aux)

        return up

    def compute_p(self,tripsloader, usersbystop, service):
        if os.path.exists(RouteTrip.__COEF_BASENAME % service):
            return pd.read_csv(RouteTrip.__COEF_BASENAME % service, index_col = 'stop_id')

        timeseries_o = tripsloader.timeseries_o
        routes = tripsloader.routesLoader.routes.set_index("stop_id")

        stops = routes.loc[usersbystop[usersbystop.stop_id.notna()].stop_id]
        stops = stops[~stops.index.duplicated(keep='first')]

        coef = usersbystop.merge(stops["stop_district"], right_index=True, left_on="stop_id")

        subtimeseries_o = timeseries_o[stops.stop_district.unique()][timeseries_o.index.month == 2]
        subtimeseries_o = subtimeseries_o.between_time("6:00", "0:00")

        if service == 'cercanias':
            subtimeseries_o = subtimeseries_o[subtimeseries_o.index.weekday < 5]

        subtotal_o = subtimeseries_o.sum()

        subtotal = pd.DataFrame({"o": subtotal_o})
        coef = coef.merge(subtotal, left_on="stop_district", right_index=True)
        coef["p_o"] = coef["up"] * subtimeseries_o.index.day.unique().shape[0] / coef["o"]

        if service == 'metro':
            coef['p_o'] = coef['p_o']/29

        coef = coef.set_index("stop_id")
        coef.to_csv(RouteTrip.__COEF_BASENAME % service)

        return coef


class DataManager:
    def __init__(self,routetrip):
        self.routetrip = routetrip

        renfe_up = routetrip.get_users_renfe(routetrip.tripsLoader.timeseries_o)

        metro_up = routetrip.get_users_metro(routetrip.tripsLoader.timeseries_o,
                                                        max_precision=False)

        self.up = {'cercanias':renfe_up,
                   'metro':metro_up}
