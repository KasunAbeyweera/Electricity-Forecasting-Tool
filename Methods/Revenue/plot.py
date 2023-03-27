import plotly
import plotly.express as px
import pandas as pd
import json
from plotly.io import to_json

# def plotly():
#     df = pd.read_csv('C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Sales.csv')
#     fig = px.line(df, x=('Date'), y='totalSales', title='test')
#
#     # Create graphJSON
#     graphJSON = to_json(fig)
#
#     # Use render_template to pass graphJSON to html
#     return graphJSON
import plotly
import plotly.express as px
import pandas as pd
import json
from plotly.io import to_json

def ploytPlot():
    df = pd.read_csv('C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Sales.csv')
    fig = px.line(df, x=('Date'), y='totalSales')

    # Create graphJSON
    graphJSON = to_json(fig)
    graph1Json = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)

    # Use render_template to pass graphJSON to html
    return graph1Json

def chartJsPlot():
    df = pd.read_csv('C:\\Users\\kasun\\Downloads\\flaskProject\\Datasets\\Sales.csv')
    # Extract the date and total sales columns from the DataFrame
    data = df.to_dict(orient='list')

    # Pass the dictionary to your template
    return data
