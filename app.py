# Data Libraries
import pandas as pd
import numpy as np

# App libraries
import dash_core_components as dcc
import dash_html_components as html
#import dash_table_experiments as dt
import dash.dependencies
from dash.dependencies import Input, Output, State
# import requests

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Visualization Libraries
#import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px



## THE COMPONENTS
## GRAPHS 
def plot_dist_churn(df, col, binary='Churn'):
    tmp_churn = df[df[binary] == 1]
    tmp_no_churn = df[df[binary] == 0]
    
    trace1 = go.Bar(
        x=tmp_churn[col].value_counts().sort_index().index,
        y=tmp_churn[col].value_counts().sort_index().values,
        name='Churn',opacity = 0.8, marker=dict(
            color='turquoise',
            line=dict(color='#000000', width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col].value_counts().sort_index().index,
        y=tmp_no_churn[col].value_counts().sort_index().values,
        name='No Churn', opacity = 0.8, 
        marker=dict(
            color='coral',
            line=dict(color='#000000',
                      width=1)
        )
    )

   
    layout = dict(title =  f'Distribution of {str(col)} feature <br>by Churn',
              xaxis=dict(), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [0, 100], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False,
                         ))

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    fig.update_layout(title_x=.5, legend_orientation='h', 
                      height=450, 
                      legend=dict(x=.002, y=-.06))
    return fig

def plot_dist_churn2(df, col):

    tmp_attr = df[col].value_counts()

    trace1 = go.Bar(
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        #name='Yes_Churn',
        opacity = 0.8, marker=dict(
            color='indianred',
            line=dict(color='#000000',width=1)))
    
    layout = dict(title =  f'Distribution of {str(col)}',
                  xaxis={'type':'category'}, 
                  yaxis=dict(title= 'Count'))

    fig = go.Figure(data=[trace1], layout=layout)
    fig.update_layout(title_x=.5, legend_orientation='h', height=450,
                      legend=dict(x=.8, y=-.06))

    return fig

def pie_norm(df, val1, val2, limit=15):
    tmp = df.groupby(val1)[val2].sum().nlargest(limit).to_frame().reset_index()
    tmp = tmp.sort_values(val1)

    trace1 = go.Pie(labels=tmp[val1], sort=False, 
                    values=tmp[val2], name=str(val1), hole= .5, 
                    hoverinfo="label+percent+name+value", 
                    )

    layout = dict(title={'text':str(f"{val2} Ratio of <br>{val1} by General")},
    titlefont={'size':15}
        )
    fig  = go.Figure(data=[trace1], layout=layout)

    fig.update_layout(title_x=.5, 
                      legend_orientation='h', 
                      legend=dict(x=.003, y=.01))
    
    fig['layout']['height'] = 380
    fig['layout']['width'] = 350
    
    return fig

def pie_churn(df, val1, val2, binary, limit=15):
    
    if binary == 'Churn':
        mat = 1
    else:
        mat = 0

    tmp = df[df['Churn'] == mat].groupby(val1)[val2].sum().nlargest(limit).to_frame().reset_index()

    tmp = tmp.sort_values(val1)

    trace1 = go.Pie(labels=tmp[val1], sort=False,
                    values=tmp[val2], name=f'{binary}', hole= .5, 
                    hoverinfo="label+percent+name+value", showlegend=True,
                    #domain= {'x': [0, .48]}
                    )

    layout = dict(title={'text':str(f"{val2} Ratio of <br>{val1} by {binary}")}, 
        titlefont={'size':15})

    fig  = go.Figure(data=[trace1], layout=layout)
    fig.update_layout(title_x=.5, 
                      legend_orientation='h', 
                      legend=dict(x=.003, y=.01))
    fig['layout']['height'] = 380
    fig['layout']['width'] = 350

    return fig

####LAYOUTS

theme = {'font-family': 'Raleway', 'background-color': '#787878'}


def graph_1():
    graph = html.Div([   
            dcc.Graph(id='Graph1', 
                      className="six columns", 
                    ),
            dcc.Graph(id='Graph4', 
            className="six columns", )
            ])
    return [graph]


def graph2_3():
    graph_2 = html.Div([
                dcc.Graph(id='Graph2',
                        )], className="four columns", style={'margin':'0'})
    graph_5 = html.Div([
                dcc.Graph(id='Graph5',
                        )], className="four columns", style={'margin':'0'})
    graph_3 = html.Div([   
                dcc.Graph(id='Graph3',
                        )], className="four columns", style={'margin':'0'})

    # Return of the graphs in all the row
    return [graph_2, graph_3, graph_5]


def create_header(some_string):
    header_style = {
        'background-color':'#BF3E22',
        'padding': '1rem',
        'display':'inline-block',
        'width':'100%',
        'font-weight':'bold'
        
    }

    su = html.Img(
                    src='https://www.logolynx.com/images/logolynx/3b/3b431fa20a112a5bec8fb33c1d039a10.jpeg',
                    className='three columns',
                    style={
                        'height': 'auto',
                        'width': '120px', # 'padding': 1
                        'float': 'right', #'position': 'relative'
                        'margin-right': '20px', #'border-style': 'dotted'
                        'display':'inline-block'})

    title = html.H1(children=some_string, className='eight columns',
                    style={'margin': '25px',
                           'color':'#ffffff', 'font-size':'50px'})
    

    header = html.Header(html.Div([title,su]), style=header_style)

    return header


def create_footer():
    footer_style = {
        'font-size': '2.2rem',
        'background-color': '#BF3E22',
        #'padding': '2.5rem',
        'margin-top': '3rem', 
        'display':'inline-block', 'padding':'16px 32px 8px'
    }
    footer = html.Footer(style=footer_style, className='twelve columns')
    return footer


def header_logo():
    h1_title = html.H1(
                    children='TELECOM CUSTOMER RETENTION ANALYSIS',
                    style={
                           'color':'#ffffff',
                           'margin-top':5,
                           'font-size':'35px'
                            })
    return [h1_title]
    
def paragraphs():
    div = html.H1("DISTRIBUTION OF REVENUE", style={'width':'85%', 'color':'#ffffff',
                                          'margin':'0 auto',  
                                          'font-size':'35px',
                                          'text-align':'center','padding':'24px 0px 20px'})
   # para = html.P(dcc.Markdown("  **Revenue churn** is the monetary amount of recurring revenue lost in a period divided by the total revenue at the beginning of the period."), 
    #style={'width':'85%', 'margin':'0 auto', 
     #               'padding-bottom':'24px', 
      #                     # 'margin':'24px 0'
       #                     })

    return [div]

tabs_style = {
        'padding': '1rem',
        'display':'inline-block',
        'width':'100%'}

#### buttons
import dash_core_components as dcc
import dash_html_components as html

cat_features = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PaperlessBilling', 'PhoneService', 'Contract', 'StreamingMovies',
                'StreamingTV', 'TechSupport', 'OnlineBackup', 'OnlineSecurity',
                'InternetService', 'MultipleLines', 'DeviceProtection', 'PaymentMethod']

def button_line():

    dropdown1 = html.Div(
        [
        html.P("Categorical Variables: ", style={'font-weight':'bold'}),
        dcc.Dropdown(
            id='dropdown',
            options=[{'label': i, 'value': i} for i in cat_features],
            value='Gender',
        ),
    ], className='three columns',
        style={'margin':'20px', 'display':'inline-block'})

    dropdown2 = html.Div(
        [
        html.P("Numerical Variables: ", style={'font-weight':'bold'}),
        dcc.Dropdown(
            id='dropdown2',
            options=[{'label': 'MonthlyCharges', 'value': 'MonthlyCharges'},
                     {'label': 'TotalCharges', 'value':'TotalCharges'}] ,
            value='MonthlyCharges',
        ),
    ], className='three columns',
        style={'margin':'20px', 'display':'inline-block'}
    )
    button = html.Div(
        [
        html.P("Type of Distribution: ", style={'font-weight':'bold'}),
        dcc.RadioItems(
            id='churn-or-not',
            options=[{'label': 'Based on Churn', 'value':'Churn'}, 
                     {'label': 'General', 'value': 'Normal'},
                     ], labelStyle={'display':'inline-block',
                     } ,
            value='Churn',
        ), 
      ], className='three columns', style={'margin-top':'20px', 'display':'inline-block'}
    )

    return [dropdown1, dropdown2, button]

def lg():
    lg_penalty = html.Div([
                           html.P("penalty",style={'font-weight':'bold'}),
                           dcc.Dropdown(
                                   id='lg_penalty',
                                   options=[{'label': 'None', 'value':'None'},
                                            {'label':'l2','value':'l2'}],
                                            value='l2',
                    ),],
                 className='three columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
    lg_c = html.Div([
                        html.P('C',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='lg_c',
                                options=[{'label':'30','value':30},
                                         {'label':'40','value':40},
                                         {'label':'50','value':50},
                                         {'label':'60','value':60}],
                                         value=50,),],
                                                 className='three columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
    
    return [lg_c, lg_penalty]

def rf():
    rf_max_depth = html.Div([
                           html.P("max Depth",style={'font-weight':'bold'}),
                           dcc.Dropdown(
                                   id='rf_md',
                                   options=[{'label':'1','value':1},
                                            {'label':'3','value':3},
                                            {'label':'6','value':6},
                                            {'label':'8','value':8}],
                                            value='3',
                    ),],
                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
    rf_criterion = html.Div([
                        html.P('criterion',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='rf_c',
                                options=[{'label':'gini','value':'gini'},
                                         {'label':'entropy','value':'entropy'}],
                                         value='entropy',),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )     
    rf_min_s_split = html.Div([
                        html.P('min samples split',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='rf_mss',
                                options=[{'label':'10','value':10},
                                         {'label':'12','value':12},
                                         {'label':'14','value':14},
                                         {'label':'16','value':16}],
                                         value='12',),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
                                                                           
    rf_min_s_leaf = html.Div([
                        html.P('min samples leaf',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='rf_msl',
                                options=[{'label':'1','value':1},
                                         {'label':'2','value':2},
                                         {'label':'3','value':3},
                                         {'label':'5','value':5}],
                                         value='2',),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
    
    rf_est = html.Div([
                        html.P('num of estimators',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='rf_est',
                                options=[{'label':'100','value':100},
                                         {'label':'200','value':200},
                                         {'label':'300','value':300},
                                         {'label':'400','value':400}],
                                         value='100',),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
    
    
    
    return [rf_criterion,rf_max_depth,rf_min_s_leaf,rf_min_s_split,rf_est] 


def gbm():
    gbm_max_depth = html.Div([
                           html.P("max Depth",style={'font-weight':'bold'}),
                           dcc.Dropdown(
                                   id='gbm_md',
                                   options=[{'label':'2','value':2},
                                            {'label':'3','value':3},
                                            {'label':'4','value':4},
                                            {'label':'5','value':5}],
                                            value='3',
                    ),],
                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
    gbm_learning = html.Div([
                        html.P('learning rate',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='gbm_lr',
                                options=[{'label':'0.02','value':0.02},
                                         {'label':'0.06','value':0.06},
                                         {'label':'0.1','value':0.1}],
                                         value=0.02,),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )     
    gbm_est = html.Div([
                        html.P('num of estimators',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='gbm_est',
                                options=[{'label':'400','value':40},
                                         {'label':'600','value':600},
                                         {'label':'800','value':800},
                                         {'label':'1000','value':1000}],
                                         value=600,),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
                                                                  
    return [gbm_max_depth,gbm_learning,gbm_est]

def lgbm():
    lgbm_learning = html.Div([
                        html.P('learning rate',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='lgbm_lr',
                                options=[{'label':'0.02','value':0.02},
                                         {'label':'0.06','value':0.06},
                                         {'label':'0.1','value':0.1}],
                                         value=0.02,),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )     
    lgbm_est = html.Div([
                        html.P('num of estimators',style={'font-weight':'bold'}),
                        dcc.Dropdown(
                                id='lgbm_est',
                                options=[{'label':'400','value':40},
                                         {'label':'600','value':600},
                                         {'label':'800','value':800},
                                         {'label':'1000','value':1000}],
                                         value=600,),],
                                                 className='two columns', style={'background': '#f77754', 'padding':'15px 24px', 'margin':'0px 24px'} )
                                                                  
    return [lgbm_learning,lgbm_est]
#load_lg = pickle.load(open(Logistic.sav, 'rb'))
    
## CSS EXTERNAL FILE
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 
                        'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
                        'https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css']

## Importing the dataset
df = pd.read_csv('telco.csv')

## Some initial modification in the data
df.drop('customerID',axis=1,inplace=True)
df.rename(columns = {'gender':'Gender','tenure':'Tenure'},inplace=True)
df['TotalCharges'] = df['TotalCharges'].replace(' ',np.nan)
df.dropna(inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
df['Churn_label'] = df.Churn.copy()
df['Churn'] = df.Churn.replace({'Yes': 1, 'No': 0})

X = df.drop(['Churn'],axis=1)
y=df['Churn']

X = pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaled_X_train = StandardScaler().fit_transform(X_train)
scaled_X_test = StandardScaler().fit_transform(X_test)


## App Name
app_name='IST 707 DATA ANALYTICS'

# Instantiating our app
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)

# Seting the name to app
app.title=app_name

server = app.server

def tab_1():
    tab1 = html.Div([
        html.Div(header_logo(), className='row', style={'text-align': 'center', 'background':'#008B8B', 
                                                        'margin': '24px 0', 'padding':'24px 24px 10px 24px'}
        ), ## Title and logo inline
        #html.Div(paragraph_header(), className='row'),
        html.Div(button_line(), className='row', style={'background': '#f77754', 'padding':'0px 24px', 'margin':'24px 0'}),
        html.Div(graph_1(), className='row', style={'padding-top':'10'}), # first and principal graph
        html.Div(paragraphs(), className='row', style={'background': '#008B8B', 'padding':'0px 0px', 'margin':'24px 0'}), 
        html.Div(graph2_3(), className='row') # Pie graphs
    ],className='container'    # style={'width':'85%', 'margin':'0 auto'}
    )      # setting the class container to become all in a "box" on the browser. Only header and footer will be out of it
    return tab1

def tab_2():
    tab2 = html.Div([
            html.Div(lg(),className='row', style={'text-align': 'center', 
                                                        'margin': '18px 0', 'padding':'24px 24px 10px 24px'}),
            
          
            ])
    return tab2

def tab_3():
    tab3 = html.Div([
            html.Div(rf(),className='row', style={'text-align': 'center', 
                                                        'margin': '18px 0', 'padding':'24px 24px 10px 24px'})])
    return tab3

def tab_4():
    tab4 = html.Div([
            html.Div(gbm(),className='row', style={'text-align': 'center', 
                                                        'margin': '18px 0', 'padding':'24px 24px 10px 24px'})])
    return tab4

def tab_5():
    tab5 = html.Div([
            html.Div(lgbm(),className='row', style={'text-align': 'center', 
                                                        'margin': '18px 0', 'padding':'24px 24px 10px 24px'})])
    return tab5
# main APP engine
app.layout = html.Div([create_header(app_name),
        dcc.Tabs([
                dcc.Tab(label='VISUALIZATION',style={'borderBottom':'1px solid #d6d6d6',
                                                     'padding':'10px',
                                                     'fontWeight':'bold',
                                                     'font-size':'15px'},selected_style={
                                                                                    'borderTop': '1px solid #d6d6d6',
                                                                                    'borderBottom': '1px solid #d6d6d6',
                                                                                    'backgroundColor': '#119DFF',
                                                                                    'color': 'white',
                                                                                    'padding': '10px',
                                                                                    'font-size':'15px',
                                                                                    'fontWeight':'bold'},
    children=[ 
                html.Div(children=[ 
                tab_1(),
                html.Div(create_footer())],
                                          style={'overflow':'hidden'})]),
    
                dcc.Tab(label='LOGISTIC REGRESSION',style={'borderBottom':'1px solid #d6d6d6',
                                                     'padding':'10px',
                                                     'fontWeight':'bold',
                                                     'font-size':'15px'},selected_style={
                                                                                    'borderTop': '1px solid #d6d6d6',
                                                                                    'borderBottom': '1px solid #d6d6d6',
                                                                                    'backgroundColor': '#119DFF',
                                                                                    'color': 'white',
                                                                                    'padding': '10px',
                                                                                    'font-size':'15px',
                                                                                    'fontWeight':'bold'},
    children=[
                html.Div(children=[
                        tab_2()
                        
                        ])]),
    
                dcc.Tab(label='RANDOM FOREST',style={'borderBottom':'1px solid #d6d6d6',
                                                     'padding':'10px',
                                                     'fontWeight':'bold',
                                                     'font-size':'15px'},selected_style={
                                                                                    'borderTop': '1px solid #d6d6d6',
                                                                                    'borderBottom': '1px solid #d6d6d6',
                                                                                    'backgroundColor': '#119DFF',
                                                                                    'color': 'white',
                                                                                    'padding': '10px',
                                                                                    'font-size':'15px',
                                                                                    'fontWeight':'bold'},
    children=[
                html.Div(children=[
                        tab_3()
                    
                        ])]),  
        
                
                dcc.Tab(label='GRADIENT BOOSTING',style={'borderBottom':'1px solid #d6d6d6',
                                                     'padding':'10px',
                                                     'fontWeight':'bold',
                                                     'font-size':'15px'},selected_style={
                                                                                    'borderTop': '1px solid #d6d6d6',
                                                                                    'borderBottom': '1px solid #d6d6d6',
                                                                                    'backgroundColor': '#119DFF',
                                                                                    'color': 'white',
                                                                                    'padding': '10px',
                                                                                    'font-size':'15px',
                                                                                    'fontWeight':'bold'},
    children=[
                html.Div(children=[
                        tab_4()
                    
                        ])]),  
                
                dcc.Tab(label='LIGHT GRADIENT BOOSTING',style={'borderBottom':'1px solid #d6d6d6',
                                                     'padding':'10px',
                                                     'fontWeight':'bold',
                                                     'font-size':'15px'},selected_style={
                                                                                    'borderTop': '1px solid #d6d6d6',
                                                                                    'borderBottom': '1px solid #d6d6d6',
                                                                                    'backgroundColor': '#119DFF',
                                                                                    'color': 'white',
                                                                                    'padding': '10px',
                                                                                    'font-size':'15px',
                                                                                    'fontWeight':'bold'},
    children=[
                html.Div(children=[
                        tab_5()
                    
                        ])]),  
    ])])
                      


###################################################
## first line of graphsGraph 1 of the first line ##
###################################################
@app.callback(
    dash.dependencies.Output('Graph1', 'figure'),
    [dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('churn-or-not', 'value')])
def binary_ploting_distributions(cat_col, binary_selected):
    from plotly import tools
    #print(binary_selected)
    if binary_selected == 'Churn':
        return plot_dist_churn(df, cat_col)
    else:
        return plot_dist_churn2(df, cat_col)
    
# 
@app.callback(
    dash.dependencies.Output('Graph2', 'figure'),
    [dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('dropdown2', 'value')])
def _graph_upgrade2(val1, val2):
    """
    This function helps to investigate the proportion of metrics of toxicity and other values
    """
    return pie_norm(df, val1, val2)

# 
@app.callback(
    dash.dependencies.Output('Graph3', 'figure'),
    [dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('dropdown2', 'value')])
def PieChart(val1, val2, limit=15):
    """
    This function helps to investigate the proportion of metrics of toxicity and other values
    """
    return pie_churn(df, val1, val2, "No-Churn")


# Graph of histogram in adressed on Graph4
@app.callback(
    dash.dependencies.Output('Graph4', 'figure'),
    [dash.dependencies.Input('dropdown2', 'value'),
     dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('churn-or-not', 'value')])
def _plotly_express(cat_col, color, churn): 
    if churn == "Churn":
        fig = px.box(df, x=color, y=cat_col, 
                    color=df['Churn_label'].map({'Yes':'Churn', 'No':'NoChurn'}), height=450,       
                    color_discrete_map={"Churn": "steelblue", 
                                        "NoChurn": "tomato"},
                    category_orders={str(color):df[color].value_counts().sort_index().index}
                )
        fig.update_layout(
            title=f"{cat_col} distribution <br>by {color} & Churn",
            xaxis_title=dict(), showlegend=True,
            yaxis_title=f"{cat_col} Distribution", 
            title_x=.5, legend_title=f'Churn:', 
            xaxis={'type':'category'},
            margin=dict(t=100, l=50)
        )
    else:
        fig = px.box(df, x=color, y=cat_col, 
                    height=450,        
                    category_orders={str(color):df[color].value_counts().sort_index().index},
                    color_discrete_sequence = ['mediumseagreen']
            )

        fig.update_layout(
            title=f"Distribution of {cat_col} <br>by {color}",
            xaxis_title=dict(), showlegend=True,
            yaxis_title=f"{cat_col} Distribution",

            title_x=.5, legend_title=f'Churn:', 
            xaxis={'type':'category'},
            margin=dict(t=100, l=50)
            )

    fig.update_xaxes(title='')

    return fig

# 
@app.callback(
    dash.dependencies.Output('Graph5', 'figure'),
    [dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('dropdown2', 'value')])
def PieChart(val1, val2, limit=15):
    """
    This function helps to investigate the proportion of metrics of toxicity and other values
    """
    return pie_churn(df, val1, val2, "Churn")


if __name__ == '__main__':
    app.run_server(debug=True)

