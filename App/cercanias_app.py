import numpy as np
import seaborn as sb
import pandas as pd
from datetime import date
import datetime
import matplotlib.pyplot as plt
from fbprophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import pickle
from datetime import datetime, timedelta

from processor import * #Importamos todas las clases del archivo processor
from data_injector import *
from forecast_system import *

import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)
import dash_bootstrap_components as dbc
from colour import Color

import ray

WINDOW_BEGIN = date(2020,3,1)
WINDOW_END = date(2020,3,31)
default_date = date(2020, 7, 19)


external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",

    },
    # dbc.themes.BOOTSTRAP
    # dbc.themes.DARKLY
    dbc.themes.SUPERHERO
]

app = Dash(__name__,external_stylesheets=external_stylesheets)
ray.init()

# app.layout = html.Div([
#     html.H1("Selecciona una fecha y hora"),
#     dcc.DatePickerSingle(
#         id='date-picker',
#         placeholder="Selecciona una fecha"
#     ),
#     dcc.Dropdown(
#         id='hour-dropdown',
#         options=[{'label': str(i).zfill(2), 'value': i} for i in range(24)],
#         placeholder="Selecciona la hora"
#     ),
#     html.Div(id='output-container')
# ])

def main_layout():
    return html.Div(children=[

                            html.Div(id='Titulo Graph',children=[
                            html.H3("Sistema de Predicción por Hora de los Días Lunes",className="display-4",
                            style={"font-size": "30px", "text-align": "center", "margin-top": "1px"})
                            ]),

                            html.Div(id='output_container', children=[],
                                        style={"font-size": "20px",'margin-top': '2px'},className="display-4"),#className='text-primary'),
                            html.Div(
                                dcc.Graph(id='predictions-graph'),
                                style={'margin-top': '20px', 'margin-left': '80px'}
                            )
                            ],
                            # className="wrapper",
                            className="wrapper text-center"
                )

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "23rem",
    "padding": "2rem 1rem",
    # "background-color": "#f8f9fa",
    # "background-color": "##343a40",
    'overflow':'scroll'
}

CONTENT_STYLE = {
    "margin-left": "23rem",
    #"margin-right": "2rem",
    #"padding": "2rem 1rem",

}

navbar = dbc.NavbarSimple(
    # children=[
    #     dbc.NavItem(dbc.NavLink("Cercanias", id = 'cercanias_button', n_clicks = 0)),
    #     dbc.NavItem(dbc.NavLink("Metro", id = 'metro_button',n_clicks= 0)),
    #     # dbc.DropdownMenu(
    #     #     children = [
    #     #         dbc.DropdownMenuItem('Información',id = 'moreinfo_button',n_clicks = 0)
    #     #     ],
    #     #     nav = True,
    #     #     in_navbar=True,
    #     #     label = 'Más',
    #     #     right = True
    #     # )
    # ],
    brand="Servicio de Transporte Público Cercanías",
    brand_href="#",
    color="primary",
    dark=True,
)

#####################################################################

sidebar = html.Div(id = 'mainsidebar',children =
    [
        html.H2("DASHBOARD", className="display-4", style={"font-size": "30px", "text-align": "center", "margin-top": "1px"}),

        html.P(
            "Panel de control", className="lead"
        ),

        html.Hr(),

        html.Div(id = 'dropdown_container',children=[
            dbc.Badge("Seleccione línea", color="primary", className="mr-2"),
            dcc.Dropdown(id='lines_dropdown',
                        options=[
                            {'label': 'C1', 'value': 'C1', 'disabled': True},
                            {'label': 'C2', 'value': 'C2', 'disabled': True},
                            {'label': 'C3', 'value': 'C3', 'disabled': True},
                            {'label': 'C4', 'value': 'C4'},
                            {'label': 'C5', 'value': 'C5', 'disabled': True},
                            {'label': 'C8', 'value': 'C8', 'disabled': True},
                            {'label': 'C10', 'value': 'C10', 'disabled': True}
                        ],
                        value='C4',
                        style={"color": "darkblue"}
            )
        ], className='lead',    #style={'margin-bottom': '1rem'} #style={'padding-bottom':'3rem'}
        ),

        # html.Hr(),
        html.Br(),

        html.Div(id = 'dropdown_container',children=[
            dbc.Badge("Seleccione estación", color="primary", className="mr-2"),
            dcc.Dropdown(id='stops_dropdown',
                        options=[
                            {'label': 'Alcobendas-ss', 'value': 'Alcobendas-ss', 'disabled': True},
                            {'label': 'Valdelasfuentes', 'value': 'Valdelasfuentes', 'disabled': True},
                            {'label': 'Univ. P. Comillas', 'value': 'Univ. P. Comillas', 'disabled': True},
                            {'label': 'Cantoblanco Univ.', 'value': 'Cantoblanco Univ.', 'disabled': True},
                            {'label': 'Fuencarral', 'value': 'Fuencarral', 'disabled': True},
                            {'label': 'Chamartín', 'value': 'Chamartín'},
                            {'label': 'N. Ministerios', 'value': 'N. Ministerios', 'disabled': True},
                            {'label': 'Sol', 'value': 'Sol', 'disabled': True},
                            {'label': 'Atocha', 'value': 'Atocha', 'disabled': True},
                            {'label': 'Villaverde Bajo', 'value': 'Villaverde Bajo', 'disabled': True},
                            {'label': 'Villaverde Alto', 'value': 'Villaverde Alto', 'disabled': True},
                            {'label': 'Las Margaritas Univ.', 'value': 'Las Margaritas Univ.', 'disabled': True},
                            {'label': 'Getafe Centro', 'value': 'Getafe Centro', 'disabled': True},
                            {'label': 'Getafe Sector 3', 'value': 'Getafe Sector 3', 'disabled': True},
                            {'label': 'Parla', 'value': 'Parla', 'disabled': True}

                        ],
                        value='Chamartín',
                        style={"color": "darkblue"}
            )
        ], className='lead', style={'margin-bottom': '1rem'} #style={'padding-bottom':'3rem'}
        ),

        html.Hr(),

        html.Div(id='badge_container',style={'text-align': 'center'},children=[
            dbc.Badge("Realizar predicción",color="danger",className="mr-2",style={"font-size": "18px", "display": "inline-block"})
            ]
        ),

        html.Div(children = [
            dbc.Badge("Seleccione fecha", color="primary", className="mr-2"),
            html.Br(),
            dcc.DatePickerSingle(
                id='date-picker',
                placeholder="Fecha",
                # initial_visible_month=default_date.strftime('%Y-%m'),
                date=default_date
            )

        ], className='lead',style={'padding-bottom':'1rem'}
        ),

        html.Div(children = [
            dbc.Badge("Seleccione hora", color="primary", className="mr-2"),
            dcc.Dropdown(
                id='hour-dropdown',
                options=[{'label': str(i).zfill(2), 'value': i} for i in range(24)],
                placeholder="Hora",
                value=23,
                style={"color": "darkblue"}

            )

        ], className='lead',style={'padding-bottom':'1rem'}
        ),

    ],
    style=SIDEBAR_STYLE,
)

#######################################################################

content = html.Div(id="main-content", style=CONTENT_STYLE, children=[navbar,
                                                                     main_layout()])



app.layout = html.Div(
    children=[
        sidebar,
        content,
        dcc.Interval(id = 'interval', interval = 1000, disabled=True,n_intervals = 0)
    ]
)




# @app.callback(
#     dash.dependencies.Output('output-container', 'children'),
#     [dash.dependencies.Input('date-picker', 'date'),
#      dash.dependencies.Input('hour-dropdown', 'value')])
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output('predictions-graph','figure')],# Mi gráfica

    [Input(component_id='date-picker', component_property='date'),
    Input(component_id='hour-dropdown', component_property='value')]
)



def cercanias(date,value):
    fecha_str = date
    hora = value
    fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
    end_date = datetime(fecha.year, fecha.month, fecha.day, hora, 0, 0)
    target_date = datetime(2020, 7, 19, 23, 0, 0)
    texto = "Estimación de las Entradas en la Estación de Charmartín"
    # texto = "Estimación hitorica de las Entradas en la Estación de Charmartín"

    global cont
    global subsetN
    global injector
    global timeseries_o
    fila=0
    start = False
    start2 = False
    start3 = False
    start4 = False
    m = None
    lista_Lunes = []

    # Crear dataframe "future"
    future = pd.DataFrame({'ds': pd.date_range(start=datetime(2020, 6, 21), end=end_date, freq='H')})

    if start == True and end_date.hour == 0:
        start = False
    elif start2 == True and end_date.hour == 6:
        start2 = False
    elif start3 == True and end_date.hour == 12:
        start3 = False
    elif start4 == True and end_date.hour == 18:
        start4 = False
    else:
        pass

    # Condición de prueba
    if cont == 0:        
        tripsloader = TripsLoader(verbose = True)
        timeseries_o = tripsloader.timeseries_o
        timeseries_o = timeseries_o.loc['2020-07-01':'2020-07-31 23:00:00']

        Trayectos = tripsloader.trayectos
        trayectos = ray.get(Trayectos)
        trayectos = trayectos.loc['2020-06-21':'2020-07-31 23:00:00']
        columnas_o = [columna for columna in trayectos.columns if columna.startswith('2807905')]
        subsetN = trayectos[columnas_o]
        subsetNn = subsetN.copy()

        injector = Injector()
        dfFinal0_5 = injector.dfFinal0_5
        dfFinal6_11 = injector.dfFinal6_11
        dfFinal12_17 = injector.dfFinal12_17
        dfFinal18_23 = injector.dfFinal18_23
    else:
        subsetNn = subsetN.copy()

    StopForecast = Forecast()
    future, m = StopForecast.start_model(future,end_date,injector,start,start2,start3,start4,fila,m)
    forecast = m.predict(future)
    prediction = forecast.iloc[[-1]].copy()
    ds = prediction.loc[:, 'ds']
    yhat = prediction.loc[:, 'yhat']
    prediction = pd.DataFrame({'ds': ds, 'yhat': yhat})
    prediction = prediction.set_index('ds')
    # print(prediction)

    Chamartin_up_Renfe = StopForecast.Input_Estimate_Cercanias(prediction,subsetNn,timeseries_o,lista_Lunes)
    # json_data = Chamartin_up_Renfe.to_json(orient='records')


    # Chamartin_up_Renfe['index_str'] = Chamartin_up_Renfe.index.strftime('%Y-%m-%d %H:%M:%S')
    if end_date == target_date:
        Chamartin_up_Renfe = Chamartin_up_Renfe.drop(Chamartin_up_Renfe.index[0])
    else:
        pass
    # Chamartin_up_Renfe = Chamartin_up_Renfe.reset_index()
    #
    # Chamartin_up_Renfe['rounded_y'] = Chamartin_up_Renfe['y'].round(1).astype(float)


    def Copy_y(Fecha):
        newFecha = "'" + str(Fecha) + "'"
        return (newFecha)

    Chamartin_prueba = Chamartin_up_Renfe.copy()
    Chamartin_prueba['Fechas'] = Chamartin_prueba.index.strftime('%Y-%m-%d %H:%M:%S')
    Chamartin_prueba = Chamartin_prueba.reset_index()

    Chamartin_prueba['newDate'] = Chamartin_prueba['Fechas'].apply(Copy_y)

    nuevo_df = pd.DataFrame(Chamartin_prueba['newDate'].values, columns=['Fechas'])
    nuevo_df['y'] = Chamartin_prueba['y']
    # nuevo_df['round_y'] = Chamartin_prueba['rounded_y']
    json_data = nuevo_df.to_json(orient='records')

    # fig = px.bar(Chamartin_up_Renfe, x='index', y='y', text='rounded_y',
    #              title='Predicciones por hora de los días lunes')
    fig = px.bar(nuevo_df, x='Fechas', y='y', text='y')

    fig.update_traces(textfont_size=12, textangle=0, texttemplate='%{text:.1f}', hovertemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
    xaxis_title='Fecha y Hora',  # Título del eje x
    yaxis_title='Número de entradas',  # Título del eje y
    uniformtext_minsize=12,
    # uniformtext_mode='hide',
    width=1200,
    height=600,
    xaxis=dict(
        tickangle=25  # Desactiva la inclinación de los títulos del eje x
    )
    )
    cont += 1

    return [texto],fig#{'data': [trace], 'layout': layout}

cont = 0
if __name__ == '__main__':
    app.run_server(debug=True)
