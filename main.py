# importing the libraries
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import numpy as  np
import plotly.graph_objs as go
from dash.dependencies import Output, Input, State

#global variables
app = dash.Dash(__name__)
model = None
sc = None

# importing the dataset
df_train = pd.read_csv("./data/train.csv")

# sample data display
def dataset_head_display():
    df_train = pd.read_csv("./data/train.csv")
    return df_train.head()

# correlation plot
def correlation_plot():
    df_train = pd.read_csv("./data/train.csv")
    fig = go.Figure(
        go.Heatmap(
            x=list(df_train.columns),
            y = list(df_train.columns),
            z=df_train.corr(),
            colorscale="rdbu",
        )
    )
    fig.update_layout(
        height=500,
        width=500,
        margin=dict(
            pad=2
        ),
        paper_bgcolor="white",
        title="Correlation between variables"    
    )
    return fig

@app.callback(
    Output("scatter-fig","figure"),
    [Input("scatter-dropdown","value")]
)
def update_scatter_graph(column):
    fig = px.scatter(df_train,x=column,y='price_range')
    return fig.update_layout(
        height=460,
        width=600,
    )

@app.callback(
    Output("outlier-graph","figure"),
    [Input("outlier-dropdown","value")]
)
def update_boxplot(value):
    fig = px.box(df_train,y=value)
    return fig

@app.callback(
    Output("dist-graph","figure"),
    [Input("dist-dropdown","value")]
)
def update_distplot(value):
    fig = px.histogram(x=df_train[value],color_discrete_sequence=["orange"],nbins=50)
    fig.update_layout(
        title="Variable Distribution",
        plot_bgcolor="white",
        xaxis_title=value,
        showlegend=False
    )
    return fig

@app.callback(
    Output("classifier-acc","children"),
    [Input("classifier-radio","value"),Input('ntrees','value'),Input('nknn','value')]
)
def update_model(value,ntrees,nknn):

    x = df_train.iloc[:,:-1]
    y = df_train.iloc[:,-1]
    global model

    from sklearn.preprocessing import StandardScaler
    global sc
    sc = StandardScaler()
    x = sc.fit_transform(x)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.25)

    if value == "LR":
        from sklearn.linear_model import LogisticRegression
        lrc = LogisticRegression()
        lrc.fit(x_train,y_train)
        y_test_pred = lrc.predict(x_test)
        model = lrc
        return f"Accuracy is {(y_test_pred == y_test).mean()*100}%"
    elif value == "DTC":
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(criterion="gini")
        dtc.fit(x_train,y_train)
        y_test_pred = dtc.predict(x_test)
        model = dtc
        return f"Accuracy is {(y_test_pred == y_test).mean()*100}%"
    elif value == "RFC":
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(n_estimators=ntrees,random_state=1123133,max_depth=8)
        rfc.fit(x_train,y_train)
        y_test_pred = rfc.predict(x_test)
        model = rfc
        return f"Accuracy is {(y_test_pred == y_test).mean()*100}%"
    else:
        from sklearn.neighbors import KNeighborsClassifier
        knnc = KNeighborsClassifier(n_neighbors=nknn)
        knnc.fit(x_train,y_train)
        y_test_pred = knnc.predict(x_test)
        model = knnc
        return f"Accuracy is {(y_test_pred == y_test).mean()*100}%"



@app.callback(
    Output("prediction-Out",'children'),
    [Input("batpower","value"),Input("blue","value"),Input("clock-speed","value"),
        Input("dual-sim","value"),Input("fc","value"),Input("fourg","value"),
        Input("intmem","value"),Input("mdep","value"),Input("mobwt","value"),
        Input("ncore","value"),Input("pc","value"),Input("pixheight","value"),
        Input("pixwidth","value"),Input("ram","value"),Input("sch","value"),
        Input("scw","value"),Input("tt","value"),Input("threeg","value"),
        Input("touscr","value"),Input("wifi","value")
    ]

)
def prediction(batpow,blue,clocksp,dual,fc,fourg,intmem,mdep,mobwt,ncore,pc,pixh,pixw,
ram,sch,scw,tt,threeg,touscr,wifi
):
    global model
    if model != None:
        try:
            x_pred = np.array([batpow,blue,clocksp,dual,fc,fourg,intmem,mdep,mobwt,ncore,
                pc,pixh,pixw,ram,sch,scw,tt,threeg,touscr,wifi
            ]).reshape(-1,20)
            global sc
            x_pred = sc.transform(x_pred)
            pred = model.predict(x_pred)
            return f"The mobile belongs to the price range category: {pred[0]}"
        except:
            return f"Enter all Inputs"
    else:
        return f"Enter all Fields"

# layout
app.layout = html.Div([

    html.H1("Mobile Price Range Prediction",id="header"),

    html.Div([
        
        html.H3("Sample Table"),
        dash_table.DataTable(
        id="datatable",
        columns=[{"name":i, "id":i} for i in dataset_head_display().columns],
        data=dataset_head_display().to_dict('records'),
        style_table={'overflowX': 'auto'}
        ),
        
    ],className="sample-table-container"),

    html.Div([
        html.Div([
            dcc.Graph(id="correlation_plot",figure=correlation_plot()),
        ],className="correlation-plot-container"),

        html.Div([
            dcc.Dropdown(
                id="scatter-dropdown",
                options=[
                    {"label":i, "value":i } for i in list(dataset_head_display().columns)[:-1]
                ],
                value="battery_power"
            ),
            dcc.Graph(
                id="scatter-fig",
            )
        ],className="scat-with-range")
    ],className="scattervar"),


    html.Div([

        html.H3(children="Outlier",id="box-h3"),
        dcc.Dropdown(
            id="outlier-dropdown",
            options=[{"label":i, "value":i} for i in ["battery_power","clock_speed","int_memory","mobile_wt","px_height","px_width","talk_time","ram"]],
            value="battery_power"
        ),
        dcc.Graph(
            id="outlier-graph"
        )

    ],className="outlier-container"),

    html.Div([

        html.H3(children="Distribution",id="dist-h3"),
        dcc.Dropdown(
            id="dist-dropdown",
            options=[{"label":i, "value":i} for i in list(df_train.columns)],
            value="battery_power"
        ),
        dcc.Graph(
            id="dist-graph"
        )

    ],className="dist-container"),


    html.Div([
        html.H3(children="Choose the Classifier",id="classifier-h3"),
        dcc.RadioItems(
            options=[
                {"label":"Logistic Regression", "value":"LR"},
                {"label":"Decision Tree Classifier", "value":"DTC"},
                {"label":"Random Forest Classifier", "value":"RFC"},
                {"label":"K Nearest Neighbors", "value":"KNN"}
            ],
            value="LR",
            id="classifier-radio"
        ),
        html.P(id="classifier-acc")

    ],className="classifier-div"),

    html.Div([
        html.Div([
            html.H3(children="Choose Model Parameters",id='parameter-h3'),

            html.P("Random Forest (N Trees)"),
            dcc.Slider(
                min=1,
                max=100,
                value=1,
                step=1,
                marks={i:f'{i}' for i in range(1,101,10)},
                id="ntrees"
            ),

            html.P("KNN (N)"),
            dcc.Slider(
                min=1,
                max=25,
                value=1,
                step=2,
                marks={i:f'{i}' for i in range(1,25,2)},
                id="nknn"
            ),
            html.H3("Prediction"),
            html.P(id="prediction-Out")
        ],className="parameter-div"),
    
        html.Div([

            html.H3(children="Make Predictions",id='prediction-header'),
            dcc.Input(
                type="number",
                placeholder="Battery Power",
                id='batpower'
            ),
            dcc.Input(
                type="number",
                placeholder="Blue",
                id='blue'
            ),
            dcc.Input(
                type="number",
                placeholder="Clock Speed",
                id='clock-speed'
            ),
            dcc.Input(
                type="number",
                placeholder="Dual Sim",
                id='dual-sim'
            ),
            dcc.Input(
                type="number",
                placeholder="FC",
                id='fc'
            ),
            dcc.Input(
                type="number",
                placeholder="4g",
                id='fourg'
            ),
            dcc.Input(
                type="number",
                placeholder="int_memory",
                id='intmem'
            ),
            dcc.Input(
                type="number",
                placeholder="M dep",
                id='mdep'
            ),
            dcc.Input(
                type="number",
                placeholder="Mobile weight",
                id='mobwt'
            ),
            dcc.Input(
                type="number",
                placeholder="N Cores",
                id='ncore'
            ),
            dcc.Input(
                type="number",
                placeholder="PC",
                id='pc'
            ),
            dcc.Input(
                type="number",
                placeholder="Pixel Height",
                id='pixheight'
            ),
            dcc.Input(
                type="number",
                placeholder="Pixel Width",
                id='pixwidth'
            ),
            dcc.Input(
                type="number",
                placeholder="RAM",
                id='ram'
            ),
            dcc.Input(
                type="number",
                placeholder="SC_H",
                id='sch'
            ),
            dcc.Input(
                type="number",
                placeholder="SC_W",
                id='scw'
            ),
            dcc.Input(
                type="number",
                placeholder="Talk Time",
                id='tt'
            ),
            dcc.Input(
                type="number",
                placeholder="3G",
                id='threeg'
            ),
            dcc.Input(
                type="number",
                placeholder="Touch Screen",
                id='touscr'
            ),
            dcc.Input(
                type="number",
                placeholder="wifi",
                id='wifi'
            )

        ],className='predict-container')

    ],className='para-pred-container')
    
],className="body-container")





if __name__ == "__main__":
    app.run_server(debug=True)