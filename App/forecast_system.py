import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fbprophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from processor import *

class Forecast:

    def est_Lunes0_5(self,ds,fila,valor):
        date = pd.to_datetime(ds)
        if (date.dayofweek == 0 and date.hour == 0) or (date.dayofweek == 0 and date.hour == 1) or (date.dayofweek == 0 and date.hour == 2) or (date.dayofweek == 0 and date.hour == 3) or (date.dayofweek == 0 and date.hour == 4) or (date.dayofweek == 0 and date.hour == 5):
            valor = True
            fila += 1
        else:
            valor = False
            fila += 1
        return (valor)
    def Regressor0_5_FutureAccUp(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal0_5 = i.dfFinal0_5
        fechafuture = dfFinal0_5.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        print(fila)
        print(date)
        print(fechafuture)
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            print("Entro")
            valor = dfFinal0_5.loc[fila[0]]
            valor = valor['Accup']
            fila[0] += 1
        else:
            print("Cero")
            valor = 0
            fila[0] += 1
        return (valor)
    def Regressor0_5_LunesBack(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal0_5 = i.dfFinal0_5
        fechafuture = dfFinal0_5.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal0_5.loc[fila[0]]
            valor = valor['t-168Mod']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)

    #################################Lunes6_11#####################################
    def est_Lunes6_11(self,ds,fila,valor):
        date = pd.to_datetime(self,ds)
        if (date.dayofweek == 0 and date.hour == 6) or (date.dayofweek == 0 and date.hour == 7) or (date.dayofweek == 0 and date.hour == 8) or (date.dayofweek == 0 and date.hour == 9) or (date.dayofweek == 0 and date.hour == 10) or (date.dayofweek == 0 and date.hour == 11):
            valor = True
            fila += 1
        else:
            valor = False
            fila += 1
        return (valor)
    def Regressor6_11_FutureAccUp(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal6_11 = i.dfFinal6_11
        fechafuture = dfFinal6_11.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal6_11.loc[fila[0]]
            valor = valor['Accup']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)
    def Regressor6_11_LunesBack(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal6_11 = i.dfFinal6_11
        fechafuture = dfFinal6_11.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal6_11.loc[fila[0]]
            valor = valor['t-168Mod']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)

    ##################################Lunes12_17####################################
    def est_Lunes12_17(self,ds,fila,valor):
        date = pd.to_datetime(ds)
        if (date.dayofweek == 0 and date.hour == 12) or (date.dayofweek == 0 and date.hour == 13) or (date.dayofweek == 0 and date.hour == 14) or (date.dayofweek == 0 and date.hour == 15) or (date.dayofweek == 0 and date.hour == 16) or (date.dayofweek == 0 and date.hour == 17):
            valor = True
            fila += 1
        else:
            valor = False
            fila += 1
        return (valor)
    def Regressor12_17_FutureAccUp(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal12_17 = i.dfFinal12_17
        fechafuture = dfFinal12_17.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal12_17.loc[fila[0]]
            valor = valor['Accup']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)
    def Regressor12_17_LunesBack(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal12_17 = i.dfFinal12_17
        fechafuture = dfFinal12_17.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal12_17.loc[fila[0]]
            valor = valor['t-168Mod']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)

    ####################################Lunes18_23##################################
    def Regressor18_23_FutureAccUp(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal18_23 = i.dfFinal18_23
        fechafuture = dfFinal18_23.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal18_23.loc[fila[0]]
            valor = valor['Accup']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)
    def Regressor18_23_LunesBack(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal18_23 = i.dfFinal18_23
        fechafuture = dfFinal18_23.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal18_23.loc[fila[0]]
            valor = valor['t-168Mod']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)

    ####################################################################################
    def Add_y(self,ds,i,fila,valor):
        date = pd.to_datetime(ds)
        dfFinal18_23 = i.dfFinal18_23
        fechafuture = dfFinal18_23.ds.loc[fila[0]]
        fechafuture = datetime.strptime(fechafuture, "%Y-%m-%d %H:%M:%S")
        if (date.month == fechafuture.month) and (date.day == fechafuture.day) and (date.year == fechafuture.year) and (date.hour == fechafuture.hour):
            valor = dfFinal18_23.loc[fila[0]]
            valor = valor['y']
            fila[0] += 1
        else:
            valor = 0
            fila[0] += 1
        return (valor)

    ########################################################3
    def Verify_Lunes(self,fechas,lista_Lunes):
        for fecha in fechas:
            if fecha.weekday() == 0:
                lista_Lunes.append(fecha)
            else:
               pass
        return (lista_Lunes)

    #####################################################
    def warm_start_params(self,m):
        """
        Retrieve parameters from a trained model in the format used to initialize a new Stan model.
        Note that the new Stan model must have these same settings:
            n_changepoints, seasonality features, mcmc sampling
        for the retrieved parameters to be valid for the new model.

        Parameters
        ----------
        m: A trained model of the Prophet class.

        Returns
        -------
        A Dictionary containing retrieved parameters of m.
        """
        res = {}
        for pname in ['k', 'm', 'sigma_obs']:
            if m.mcmc_samples == 0:
                res[pname] = m.params[pname][0][0]
            else:
                res[pname] = np.mean(m.params[pname])
        for pname in ['delta', 'beta']:
            if m.mcmc_samples == 0:
                res[pname] = m.params[pname][0]
            else:
                res[pname] = np.mean(m.params[pname], axis=0)
        return res


    def start_model(self,future,end_date,injector,start,start2,start3,start4,fila,m):
        if end_date.hour in [0, 1, 2, 3, 4, 5]:
            fila = 0
            valor=0
            # future['Lunes0_5'] = future['ds'].apply(self.est_Lunes0_5)
            future['Lunes0_5'] = future['ds'].apply(lambda ds: self.est_Lunes0_5(ds, fila, valor))
            fila = [0]
            valor=0
            # future['Accup'] = future['ds'].apply(self.Regressor0_5_FutureAccUp,i=injector)
            future['Accup'] = future['ds'].apply(lambda ds: self.Regressor0_5_FutureAccUp(ds,injector,fila,valor))

            fila = [0]
            valor=0
            # future['t-168Mod'] = future['ds'].apply(self.Regressor0_5_LunesBack,i=injector)
            future['t-168Mod'] = future['ds'].apply(lambda ds: self.Regressor0_5_LunesBack(ds,injector,fila,valor))

            print("Lunes0_5",future)

            if start == False:
                # Load model
                with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes0_5.json', 'r') as fin:
                    m = model_from_json(fin.read())
                start = True
            else:
                df = future.iloc[:-1].copy()
                fila = [0]
                valor = 0
                # df['y'] = df['ds'].apply(self.Add_y,i=injector)
                df['y'] = df['ds'].apply(lambda ds: self.Add_y(ds, injector, fila, valor))
                m2 = Prophet(changepoint_range=0.85,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                        seasonality_mode='additive',changepoint_prior_scale=1.7)
                m2.add_seasonality(name='Weekly', period=7, fourier_order=2,prior_scale=0.8)
                m2.add_seasonality(name='Daily', period=1, fourier_order=4,prior_scale=0.0064)
                m2.add_seasonality(name='Lunes0_5', period=1/4, fourier_order=6, condition_name='Lunes0_5',prior_scale=0.005)
                m2.add_regressor('t-168Mod',mode='additive',prior_scale=0.5)
                m2.add_regressor('Accup',mode='additive',prior_scale=0.05)
                m = m2.fit(df, init=warm_start_params(m))

        elif end_date.hour in [6, 7, 8, 9, 10, 11]:
            fila = 0
            valor = 0
            # future['Lunes6_11'] = future['ds'].apply(self.est_Lunes6_11)
            future['Lunes6_11'] = future['ds'].apply(lambda ds: self.est_Lunes6_11(ds, fila, valor))
            fila = [0]
            valor = 0
            # future['Accup'] = future['ds'].apply(self.Regressor6_11_FutureAccUp,i=injector)
            future['Accup'] = future['ds'].apply(lambda ds: self.Regressor6_11_FutureAccUp(ds,injector,fila,valor))
            fila = [0]
            valor = 0
            # future['t-168Mod'] = future['ds'].apply(self.Regressor6_11_LunesBack,i=injector)
            future['t-168Mod'] = future['ds'].apply(lambda ds: self.Regressor6_11_LunesBack(ds,injector,fila,valor))

            if start2 == False:
                # Load model
                with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes6_11.json', 'r') as fin:
                    m = model_from_json(fin.read())
                start2 = True
            else:
                df = future.iloc[:-1].copy()
                fila = [0]
                valor = 0
                # df['y'] = df['ds'].apply(self.Add_y,i=injector)
                df['y'] = df['ds'].apply(lambda ds: self.Add_y(ds, injector, fila, valor))
                m2 = Prophet(changepoint_range=0.8,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                        seasonality_mode='additive',changepoint_prior_scale=1)   #mcmc_samples=100  Si mejora la predicción->,n_changepoints=100
                m2.add_seasonality(name='Lunes6_11', period=1/4, fourier_order=6, condition_name='Lunes6_11',prior_scale=0.05)
                m2.add_seasonality(name='Weekly', period=7, fourier_order=2,prior_scale=0.1)
                m2.add_seasonality(name='Daily', period=1, fourier_order=4,prior_scale=0.1)
                m2.add_regressor('Accup',mode='additive',prior_scale=1,standardize=False)
                m2.add_regressor('t-168Mod',mode='additive',prior_scale=1)
                m = m2.fit(df, init=warm_start_params(m))


        elif end_date.hour in [12, 13, 14, 15, 16, 17]:
            fila = 0
            valor = 0
            # future['Lunes12_17'] = future['ds'].apply(self.est_Lunes12_17)
            future['Lunes12_17'] = future['ds'].apply(lambda ds: self.est_Lunes12_17(ds, fila, valor))
            fila = [0]
            valor = 0
            # future['Accup'] = future['ds'].apply(self.Regressor12_17_FutureAccUp,i=injector)
            future['Accup'] = future['ds'].apply(lambda ds: self.Regressor12_17_FutureAccUp(ds,injector,fila,valor))
            fila = [0]
            valor = 0
            # future['t-168Mod'] = future['ds'].apply(self.Regressor12_17_LunesBack,i=injector)
            future['t-168Mod'] = future['ds'].apply(lambda ds: self.Regressor12_17_LunesBack(ds,injector,fila,valor))

            if start3 == False:
                # Load model
                with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes12_17.json', 'r') as fin:
                    m = model_from_json(fin.read())
                start3 = True
            else:
                df = future.iloc[:-1].copy()
                fila = [0]
                valor = 0
                # df['y'] = df['ds'].apply(self.Add_y,i=injector)
                df['y'] = df['ds'].apply(lambda ds: self.Add_y(ds, injector, fila, valor))
                m2 = Prophet(changepoint_range=0.8,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                        seasonality_mode='additive',changepoint_prior_scale=0.1)
                m2.add_seasonality(name='Lunes12_17', period=1/4, fourier_order=6, condition_name='Lunes12_17',prior_scale=0.015)
                m2.add_regressor('t-168Mod',mode='additive',prior_scale=0.5)
                m2.add_regressor('Accup',mode='additive',prior_scale=0.5,standardize=False)
                m = m2.fit(df, init=warm_start_params(m))

        elif end_date.hour in [18, 19, 20, 21, 22, 23]:
            fila = [0]
            valor = 0
            # future['Accup'] = future['ds'].apply(self.Regressor18_23_FutureAccUp,i=injector)
            future['Accup'] = future['ds'].apply(lambda ds: self.Regressor18_23_FutureAccUp(ds,injector,fila,valor))
            fila = [0]
            valor = 0
            # future['t-168Mod'] = future['ds'].apply(self.Regressor18_23_LunesBack,i=injector)
            future['t-168Mod'] = future['ds'].apply(lambda ds: self.Regressor18_23_LunesBack(ds,injector,fila,valor))

            if start4 == False:
                # Load model
                with open('/home/jonathan/tesis/10mo_avance/models/serialized_model_Lunes18_23.json', 'r') as fin:
                    m = model_from_json(fin.read())
                start4 = True
            else:
                df = future.iloc[:-1].copy()
                fila = [0]
                valor = 0
                # df['y'] = df['ds'].apply(self.Add_y,i=injector)
                df['y'] = df['ds'].apply(lambda ds: self.Add_y(ds, injector, fila, valor))
                m2 = Prophet(changepoint_range=0.8,seasonality_prior_scale=0.3,weekly_seasonality=False,daily_seasonality=False,
                        seasonality_mode='additive',changepoint_prior_scale=0.1)
                m2.add_seasonality(name='Weekly', period=7, fourier_order=2,prior_scale=0.005)
                m2.add_seasonality(name='Daily', period=1, fourier_order=4,prior_scale=0.005)
                m2.add_regressor('t-168Mod',mode='additive',prior_scale=0.5)
                m2.add_regressor('Accup',mode='additive',prior_scale=0.5)
                m = m2.fit(df, init=warm_start_params(m))

        return future, m

    def Input_Estimate_Cercanias(self,prediction,subsetNn,timeseries_o,lista_Lunes):
        valores_reemplazo = prediction["yhat"].to_dict()
        valores_reemplazo

        for fecha_hora, valor in valores_reemplazo.items():
            fecha = fecha_hora.date()
            hora = fecha_hora.time().strftime('%H:%M:%S')
            subsetNn.loc[f'{fecha} {hora}', '2807905-2807901'] = valor

        columnas_o = [columna for columna in subsetNn.columns if columna.startswith('2807905')]
        subsetNn['2807905'] = subsetNn.loc[:, columnas_o].sum(axis=1)
        colum_sum = subsetNn.loc[:, '2807905']
        index = subsetNn.index
        chamartin_op = pd.DataFrame({'2807905': colum_sum})
        chamartin_op.index = index

        chamartin_op.index = pd.to_datetime(chamartin_op.index)
        subchamartin_op = chamartin_op[chamartin_op.index == fecha_hora]

        valores_reemplazo = subchamartin_op["2807905"].to_dict()

        for fecha_hora, valor in valores_reemplazo.items():
            fecha = fecha_hora.date()
            hora = fecha_hora.time().strftime('%H:%M:%S')
            timeseries_o.loc[f'{fecha} {hora}', '2807905'] = valor

        tripsloader = TripsLoader(verbose = True)
        print(2)
        ptdata = PassengersDataLoader()

        routeTrip = RouteTrip.remote(tripsloader, ptdata)

        result_renf = routeTrip.get_users_renfe.remote(timeseries_o)
        renfe_up = ray.get(result_renf)

        fecha_lunes_anterior = fecha_hora - pd.DateOffset(days=fecha_hora.dayofweek + 7)
        rango_fechas = pd.date_range(fecha_lunes_anterior, fecha_hora, freq='H')
        lista_Lunes = self.Verify_Lunes(rango_fechas,lista_Lunes)
        rango_fechas = pd.DatetimeIndex(lista_Lunes)
        rango_fechas

        Chamartin_up_Renfe = pd.DataFrame({'y': renfe_up['par_5_18']})
        Chamartin_up_Renfe = Chamartin_up_Renfe.loc[rango_fechas]

        return Chamartin_up_Renfe

    def Input_Estimate_Metro(self,prediction,subsetNn,timeseries_o,lista_Lunes):
        valores_reemplazo = prediction["yhat"].to_dict()
        valores_reemplazo

        for fecha_hora, valor in valores_reemplazo.items():
            fecha = fecha_hora.date()
            hora = fecha_hora.time().strftime('%H:%M:%S')
            subsetNn.loc[f'{fecha} {hora}', '2807905-2807901'] = valor

        columnas_o = [columna for columna in subsetNn.columns if columna.startswith('2807905')]
        subsetNn['2807905'] = subsetNn.loc[:, columnas_o].sum(axis=1)
        colum_sum = subsetNn.loc[:, '2807905']
        index = subsetNn.index
        chamartin_op = pd.DataFrame({'2807905': colum_sum})
        chamartin_op.index = index

        chamartin_op.index = pd.to_datetime(chamartin_op.index)
        subchamartin_op = chamartin_op[chamartin_op.index == fecha_hora]

        valores_reemplazo = subchamartin_op["2807905"].to_dict()

        for fecha_hora, valor in valores_reemplazo.items():
            fecha = fecha_hora.date()
            hora = fecha_hora.time().strftime('%H:%M:%S')
            timeseries_o.loc[f'{fecha} {hora}', '2807905'] = valor

        tripsloader = TripsLoader(verbose = True)
        ptdata = PassengersDataLoader()

        routeTrip = RouteTrip.remote(tripsloader, ptdata)

        result_met = routeTrip.get_users_metro.remote(timeseries_o)
        metro_up = ray.get(result_met)

        fecha_lunes_anterior = fecha_hora - pd.DateOffset(days=fecha_hora.dayofweek + 7)
        rango_fechas = pd.date_range(fecha_lunes_anterior, fecha_hora, freq='H')
        lista_Lunes = self.Verify_Lunes(rango_fechas,lista_Lunes)
        rango_fechas = pd.DatetimeIndex(lista_Lunes)
        rango_fechas

        Chamartin_up_Metro = pd.DataFrame({'y': metro_up['par_4_261']})
        Chamartin_up_Metro = Chamartin_up_Metro.loc[rango_fechas]

        return Chamartin_up_Metro
