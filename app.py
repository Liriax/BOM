# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from ete3 import Tree
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import gunicorn
from dash.dependencies import Input, Output, State

# import the classes
from RF_BOM import TreePair, TreeCompare

# style the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# define the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

tree_inputs=[]
t1 = Tree('(((1:4,2:4)a,3:1)b, (7:1,(4:3,5:1,6:1)c)d)t1;', format=1)
t2 = Tree('(((1:4,2:4)e,(3:2, 9:1)f)x, ((8:1, 5:1, 6:1)g, 7:1)h)t2;', format=1)
t3 = Tree('(((1:4,2:4)i,(3:2, 9:1)j)y, ((4:3, 5:1, 6:1)k, 3:1)l)t3;', format=1)

trees=[t2,t3]

app.layout = html.Div(
    style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'grid-gap': '2vw'},

    children=[
    html.Div(
        className="input_div", style={'margin-left': '1vw', 'margin-top': '3vw'},
        children=[
            html.H5("Die Produktvarianten als phylogenische Bäume: "),
            html.Div(children=[dcc.Input(
                    id="tree_input",
                    type="text", value='(((1:4,2:4)a,3:1)b, (7:1,(4:3,5:1,6:1)c)d)t1;'
                )]),
            html.Div(children = [
                html.H5("Andere Bäume: "), 
                html.P('(((1:4,2:4)e,(3:2, 9:1)f), ((8:1, 5:1, 6:1)g, 7:1)h);'),
                html.P('(((1:4,2:4)i,(3:2, 9:1)j), ((4:3, 5:1, 6:1)k, 3:1)l);')
                ]),
            # html.Button(id="add_new_tree_button", n_clicks=None, children="Baum hinzufügen"),
            # html.Div(id="new_trees_div"),
            # html.Button(id="show_tree_button", n_clicks=None, children="Baum zeigen"),
            html.Img(id="tree_image"),
            html.Br(),
            html.Button(id="identify_same_nodes_button", n_clicks=None, children="Identische Baugruppe identifizieren"),
            html.Button(id="identify_similar_nodes_button", n_clicks=None, children="Ähnliche Baugruppe identifizieren"),


        ]
    ),
    html.Div(
        className="output_div",
        children=[
            html.Div(
                    id="same_nodes_output"
            ),
            html.Div(
                id="similar_nodes_output"
            )
        ]
    )]
)
# @app.callback(
#     Output("new_trees_div","children"),
#     Input("add_new_tree_button", "n_clicks"),
# )
# def add_new_trees (n_clicks):
#     if n_clicks is not None:
#         tree_inputs.append(
#             html.Div(children=[
#                 dcc.Input(
#                     id="new_tree_input",
#                     type="text", value=None
#                 )
#             ])
            
#         )
#     return tree_inputs

# @app.callback(
#     Output("tree_image","src"),
#     Input("show_tree_button", "n_clicks"),
#     State("tree_input", "value")
# )
# def show_tree (n_clicks, tree_input):
#     tree = Tree(tree_input, format=1)
#     if n_clicks is not None:
#         tree.render("tree1.png", w=60, units="mm")
#         return app.get_asset_url('tree1.png')

#     return None

@app.callback(
    Output("same_nodes_output","children"),
    Input("identify_same_nodes_button", "n_clicks"),
    State("tree_input", "value"),
)
def same_nodes (n_clicks, tree_input):
    tree = Tree(tree_input, format=1)
    output=[]
    output.append(html.Pre(tree.get_ascii(show_internal=True)))
    for x in trees:
        output.append(html.Pre(x.get_ascii(show_internal=True)))
    if n_clicks is not None:
        tc = TreeCompare(tree, trees)
        for t1_tn in tc.find_same_nodes():
            for node_pair in t1_tn:
                node_t1 = node_pair[0]
                node_tn = node_pair[1]
                output.append(html.P("Baugruppe {} ist identisch zu Baugruppe {}.".format(node_t1.name, node_tn.name)))

    return output

@app.callback(
    Output("similar_nodes_output", "children"),
    Input("identify_similar_nodes_button", "n_clicks"),
    State("tree_input","value")
)
def similar_nodes (n_clicks, tree_input):
    tree = Tree(tree_input, format=1)
    output=[]
    if n_clicks is not None:
        tc = TreeCompare(tree, trees)
        output.append(str(tc.find_similar_nodes()))
    return output
# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
