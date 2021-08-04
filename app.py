# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from ete3 import Tree
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import gunicorn
from dash.dependencies import Input, Output, State
import xlsxwriter

# import the classes
from RF_BOM import TreePair, TreeCompare
from baukastenstuecklisten import formats, encode_variante, encodings
from tree_parser import seq_L1, seq_L2, seq_L3, sachn_formenschl, find_process

# style the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# define the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

options = []
for e in encodings:
    options.append({'label':e, 'value':e})


app.layout = html.Div(
    style={'display': 'grid', 'grid-template-columns': '1fr 1fr 1fr 1fr', 'grid-gap': '2vw'},

    children=[
    html.Div(
        className="input_div", style={'margin-left': '1vw', 'margin-top': '3vw'},
        children=[
            html.H5("Die Produktvarianten als phylogenische Bäume: "),
            html.Div(children=[dcc.Dropdown(
                    id="tree_input",
                    options=options, value="FU114SA1"
                )]),
            html.Div(children = [
                dcc.Checklist(
                    id = "trees_cl",
                    options=options,
                    value=encodings[1:3]
                )
                ]),
            # html.Button(id="add_new_tree_button", n_clicks=None, children="Baum hinzufügen"),
            # html.Div(id="new_trees_div"),
            # html.Button(id="show_tree_button", n_clicks=None, children="Baum zeigen"),
            html.Img(id="tree_image"),
            html.Br(),
            html.Button(id="identify_same_nodes_button", n_clicks=None, children="Identische Baugruppe identifizieren"),
            
            html.Br(),
            html.Button(id="identify_similar_nodes_button", n_clicks=None, children="Ähnliche Baugruppe identifizieren"),
            dcc.Checklist(
                    id = "similarity_cl",
                    options=[{'label':'Bauteil-Ähnlichkeit berücksichtigen','value':"y"}],
                    value=[]
            ),

            html.Button("Ergebnisse Herunterladen", id="download_button", n_clicks=None), 
            dcc.Download(id="download")



        ]
    ),
    html.Div(
        className="output_div1",style={'margin-left': '1vw', 'margin-top': '3vw'},
        children=[
            html.Div(
                    id="print_trees_output"
            ),
        ]
    ),
    html.Div(
        className="output_div2",style={'margin-left': '1vw', 'margin-top': '3vw'},
        children=[
            html.Div(
                    id="same_nodes_output"
            ),
        ]
    ),
    html.Div(
        className="output_div3",style={'margin-left': '1vw', 'margin-top': '3vw'},
        children=[
            
            html.Div(
                id="similar_nodes_output"
            ),
            html.Div(
                id="dummy_output"
            ),

        ]

    )]
)
@app.callback(
    Output("print_trees_output","children"),
    Input("tree_input", "value"),
    Input("trees_cl","value")
)
def print_trees (tree_input, trees_cl):
    tree = Tree(formats[encodings.index(tree_input)], format=1)
    tree.get_tree_root().name= tree_input

    output=[]
    output.append(html.Pre(tree.get_ascii(show_internal=True)))
    trees=[]
    for x in trees_cl:
        t = Tree(formats[encodings.index(x)], format=1)
        t.get_tree_root().name= x

        trees.append(t)
        output.append(html.Pre(t.get_ascii(show_internal=True)))
    
    return output


@app.callback(
    Output("same_nodes_output","children"),
    Input("identify_same_nodes_button", "n_clicks"),
    Input("tree_input", "value"),
    Input("trees_cl","value"),
)
def same_nodes (n_clicks, tree_input, trees_cl):
    tree = Tree(formats[encodings.index(tree_input)], format=1)
    tree.get_tree_root().name= tree_input

    output=[]
    trees=[]
    for x in trees_cl:
        t = Tree(formats[encodings.index(x)], format=1)
        t.get_tree_root().name= x
        trees.append(t)
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
    Input("tree_input","value"),
    Input("trees_cl","value"),
    Input("similarity_cl","value"),

)
def similar_nodes (n_clicks, tree_input, trees_cl, similarity_cl):
    consider_comp_similarity=True if similarity_cl==['y'] else False
    tree = Tree(formats[encodings.index(tree_input)], format=1)
    tree.get_tree_root().name= tree_input
    output=[]
    trees=[]
    for x in trees_cl:
        t = Tree(formats[encodings.index(x)], format=1)
        t.get_tree_root().name= x
        trees.append(t)
    if n_clicks is not None:
        tc = TreeCompare(tree, trees, sachn_formenschl)
        distances = tc.find_distances(consider_comp_similarity)
        for i in range(0, len(trees_cl)):
            output.append(html.P("Produktvariante {} ist {} ähnlich zu Produktvariante {}.".format(tree_input, round(distances[i],2),trees_cl[i])))
        sim_nodes_dics = tc.find_similar_nodes(consider_comp_similarity)
        for dic in sim_nodes_dics:
            for key in dic.keys():
                if key[1] != trees_cl[sim_nodes_dics.index(dic)]:
                    output.append(html.P("Baugruppe {} ist {} ähnlich zu Baugruppe {} der Produktvariante {}.".format(key[0],round( dic.get(key),3),key[1], trees_cl[sim_nodes_dics.index(dic)])))
    return output

@app.callback(
    Output("download", "data"),
    Input("download_button", "n_clicks"),
    State("tree_input","value"),
    State("trees_cl","value"),
    State("similarity_cl","value"),
    prevent_initial_call = True,
)
def download_func (n_clicks, tree_input, trees_cl,similarity_cl):
    consider_comp_similarity=True if similarity_cl==['y'] else False
    df_lst=[]
    if n_clicks!=None:
        tree = Tree(formats[encodings.index(tree_input)], format=1)
        tree.get_tree_root().name = tree_input
        output=[]
        trees=[]
        already_found = []
        for x in trees_cl:
            t = Tree(formats[encodings.index(x)], format=1)
            t.get_tree_root().name = x
            trees.append(t)
        tc = TreeCompare(tree, trees, sachn_formenschl)
        same_nodes_dict = tc.find_same_nodes()
        same_nodes=[]
        for t1_tn in same_nodes_dict:
            for node_pair in t1_tn:
                node_t1 = node_pair[0]
                node_tn = node_pair[1]
                same_nodes.append(node_pair)
        
        # first for all the nodes that have identical nodes from other trees:
        for np in list(set(same_nodes)):
            nd=np[0]
            nc=np[1]
            children_nodes = nd.children
            for cn in children_nodes:
                if nd.name != cn.name:
                    df_lst.append([nd.name, cn.name, "identisch zu {} von {}".format(nc.name, nc.get_tree_root().name), find_process([nd.name, cn.name])])
            already_found.append(nd.name)


        # then for all other nodes that do not have identical nodes from other trees:
        distances = tc.find_distances(consider_comp_similarity)
        sim_nodes_dics = tc.find_similar_nodes(consider_comp_similarity)
        for dic in sim_nodes_dics:
            for key in dic.keys():
                # key[0] is the name of the node
                # key[1] is the name of the similar node
                if key[1] != trees_cl[sim_nodes_dics.index(dic)]:
                    nd = tree.search_nodes(name=key[0])[0]
                    if nd == tree:
                        continue
                    children_nodes = nd.children
                    other_children = [x.name for x in trees[sim_nodes_dics.index(dic)].search_nodes(name=key[1])[0].children]
                    for cn in children_nodes:
                        if key[0] in already_found:
                            continue
                        if cn.name in other_children:
                            df_lst.append([nd.name, cn.name, "identisch zur ähnlichen "+ key[1]+" von " + trees_cl[sim_nodes_dics.index(dic)], find_process([nd.name, cn.name])])

                        elif nd.name != cn.name:
                            df_lst.append([nd.name, cn.name, "neue zu " + str(round( dic.get(key),3))+" ähnlich zu "+ key[1]+" von " + trees_cl[sim_nodes_dics.index(dic)], find_process([nd.name, cn.name])])
                    already_found.append(key[0])
    
      
        df = pd.DataFrame(df_lst, index = range(0, len(df_lst)), columns=["Baugrupppe","Kinder","identisch/Ähnlichkeit","verknüpfte Prozesse [Station 1, Station 2, Station 3]"])

        return dcc.send_data_frame(df.to_excel, "ergebnisse_rf_bom.xlsx", sheet_name = "Ergebnisse")
    return None





# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
