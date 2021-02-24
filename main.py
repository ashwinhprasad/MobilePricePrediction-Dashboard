# importing the libraries
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Output, Input, State

#app
app = dash.Dash(__name__)

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
            ]
        )

    ],className="classifier-div")



],className="body-container")





if __name__ == "__main__":
    app.run_server(debug=True)