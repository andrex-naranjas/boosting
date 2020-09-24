
'''
---------------------------------------------------------------
 visualization app on Dash
 Author: Jorge Salmon-Gamboa
 ---------------------------------------------------------------
'''


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], style= {'display': 'inline-block'})

def load_report(dataset):
    dataframe = pd.read_csv('output/'+ str(dataset) +'/metrics_report.csv')
    AB_time = pd.read_csv('output/' + str(dataset) + '/AdaBoostSVM_time.csv')
    return dataframe, AB_time

def load_dataset_ROC(dataset):
    # load dataframes for ROC plot analysis
    Adaroc = pd.read_csv('output/' + str(dataset) + '/Adaroc.csv')
    Graroc = pd.read_csv('output/' + str(dataset) + '/Graroc.csv')
    MLProc = pd.read_csv('output/' + str(dataset) + '/MLProc.csv')
    Ranroc = pd.read_csv('output/' + str(dataset) + '/Ranroc.csv')
    SVCroc = pd.read_csv('output/' + str(dataset) + '/SVCroc.csv')
    KNNroc = pd.read_csv('output/' + str(dataset) + '/KNeroc.csv')
    AB_SVMroc = pd.read_csv('output/' + str(dataset) + '/BoostSVM_ROC.csv')

    return Adaroc, Graroc, MLProc, Ranroc, SVCroc, AB_SVMroc, KNNroc


app.layout =  html.Div(children = [
    html.H1('AdaBoost-SVM dashboard',
        style={
            'padding' : '35px',
            'textAlign': 'center',
            'color': '#ffffff',
            'backgroundColor' : '#232428',
            'box-shadow': '3px 3px 3px grey',
            'border-radius': '15px',
            }),

    html.Div(
        html.H4('Select a dataset')
    ),
    html.Div([
        dcc.Dropdown(id='demo-dropdown', options=[
                    {'label': 'Titanic', 'value': 'titanic'},
                    {'label': 'German', 'value': 'german'},
                    {'label': 'Heart', 'value': 'heart'},
                    {'label': 'Cancer', 'value': 'cancer'},
                    {'label': 'Solar', 'value': 'solar'},
                    {'label': 'Contra', 'value': 'contra'}
                    ], style={'width': '920px'},
                value='titanic'
            ),
        html.Div(id='dd-output-container', style = {'padding': 10})
    ]),


    html.Div(id = 'plot_layout', style = {'padding' : '30px', 'text-align' : 'center'}, children = [
        html.Div(style={'padding' : '10px',
                        'box-shadow': '3px 3px 3px grey',
                        'backgroundColor': '#ffffff',
                        'border-radius': '5px'}, children = [
            html.H3('Receiver Operating Characteristic (ROC) curve'),
            dcc.Graph(
                id = 'ROC_plot',
                style={'display': 'none'})
            ]

        ),

        html.Div(id = 'table', style={'padding' : '10px',
                        'box-shadow': '3px 3px 3px grey',
                        'backgroundColor': '#ffffff',
                        'border-radius': '5px'
                        }, children =[
            html.H3('Output metrics'),
        ]),
    ], className="row"),

    html.Div(html.A(
            'Click here to see the repository on Github.',
            href='https://github.com/andrex-naranjas/boosting',
            style={
                'textAlign': 'center',
                'color': '#3399ff',
                }
            ),
        )

])

@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('plot_layout', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])

# This updates plot and table when selecting a different dataset
def update_plot(value):
    report, AB_time = load_report(value)
    Adaroc, Graroc, MLProc, Ranroc, SVCroc, AB_SVMroc, KNNroc = load_dataset_ROC(value)

    # x axis data
    Adaroc_x = Adaroc.iloc[:,0]
    Graroc_x = Graroc.iloc[:,0]
    MLProc_x = MLProc.iloc[:,0]
    Ranroc_x = Ranroc.iloc[:,0]
    SVCroc_x = SVCroc.iloc[:,0]
    KNNroc_x = KNNroc.iloc[:,0]
    AB_SVMroc_x = AB_SVMroc.iloc[:,0]

    # y axis data
    Adaroc_y = Adaroc.iloc[:,1]
    Graroc_y = Graroc.iloc[:,1]
    MLProc_y = MLProc.iloc[:,1]
    Ranroc_y = Ranroc.iloc[:,1]
    SVCroc_y = SVCroc.iloc[:,1]
    KNNroc_y = KNNroc.iloc[:,1]
    AB_SVMroc_y = AB_SVMroc.iloc[:,1]

    # The function returns Dash divs
    return [

        html.Div(style={'padding' : '10px',
                        'box-shadow': '3px 3px 3px grey',
                        'backgroundColor': '#ffffff',
                        'border-radius': '5px',
                        'display': 'inline-block',
                        'width': '50%',}, children = [
                html.H3('Receiver Operating Characteristic (ROC) curve'),
                dcc.Graph(id = 'ROC_plot',
                figure = {
                'data':[

                {'x':AB_SVMroc_x, 'y':AB_SVMroc_y, 'type': 'scatter','line': {'color': '#636efa', 'dash': 'dashdot'},
                'mode': 'lines+markers', 'name':'AB-SVM'+f'(AUC={AB_SVMroc.iloc[1,2]:.4f})'},

                {'x':Adaroc_x, 'y':Adaroc_y, 'type': 'line', 'name':'AdaBoost'+f'(AUC={Adaroc.iloc[1,2]/100:.4f})'},
                {'x':Graroc_x, 'y':Graroc_y, 'type': 'line', 'name':'XGBoost'+f'(AUC={Graroc.iloc[1,2]/100:.4f})'},
                {'x':Ranroc_x, 'y':Ranroc_y, 'type': 'line', 'name':'RandForest'+f'(AUC={Ranroc.iloc[1,2]/100:.4f})'},
                {'x':MLProc_x, 'y':MLProc_y, 'type': 'line', 'name':'NN'+f'(AUC={MLProc.iloc[1,2]/100:.4f})'},
                {'x':KNNroc_x, 'y':KNNroc_y, 'type': 'line', 'name':'KNN'+f'(AUC={KNNroc.iloc[1,2]/100:.4f})'},
                {'x':SVCroc_x, 'y':SVCroc_y, 'type': 'line', 'name':'SVC'+f'(AUC={SVCroc.iloc[1,2]/100:.4f})'},
                ],
                'layout': {'title': 'Elapsed time (fitting): '+ str(AB_time.iloc[0,0]),
                #'height': '620px',
                #'width' : '2000px',
                'xaxis' : {'title': "False positive rate"},
                'yaxis' : {'title': 'True positive rate'}
                }
                })
            ]
        ),

        html.Div(id = 'table', style={'padding' : '10px', 'text-align' : 'center',
                        'box-shadow': '3px 3px 3px grey',
                        'backgroundColor': '#ffffff',
                        'border-radius': '5px',
                        }, children =[
            html.H3('Output metrics'),
            generate_table(report)
        ])
    ]



if __name__ == '__main__':
    app.run_server(debug=True)
