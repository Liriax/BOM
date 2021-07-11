import pandas as pd
from ete3 import Tree, TreeNode, PhyloTree

xl = pd.read_excel('Baukastenstuecklisten.xlsx', header = None)
df_left = xl.iloc[:,:6]
df_right = xl.iloc[:,7:]

def split_df_by_NaN (df):
    is_NaN = df.isnull()
    row_all_NaN = is_NaN.all(axis=1)
    NaN_rows = df[row_all_NaN].index.tolist()

    dfs = []
    dfs.append(df.iloc[0:NaN_rows[0],:])
    for n in range(0, len(NaN_rows)-1):
        start = NaN_rows[n]+1
        end = NaN_rows[n+1]
        dfs.append(df.iloc[start:end, :])
    dfs.append(df.iloc[NaN_rows[len(NaN_rows)-1]+1:,:])
    return dfs

left_dfs= split_df_by_NaN(df_left)
right_dfs=split_df_by_NaN(df_right)
all_dfs = [df.rename(columns=df.iloc[1]).drop(df.index[1]) for df in left_dfs + right_dfs if df.empty == False]
dic_df = {} 
for df in all_dfs:
    dic_df[df.iloc[0,4][-6:]] = df.drop(df.index[0]).drop(columns=['Bennung'])

getriebesatz = [df.iloc[0,4][-6:] for df in all_dfs if "Getriebesatz" in str(df.iloc[0,2])] #Stufe 2, Kind von Getriebe
getriebe = [df.iloc[0,4][-6:]  for df in all_dfs if "Getriebe " in df.iloc[0,2]] #Stufe 1
motor = [df.iloc[0,4][-6:] for df in all_dfs if "Synchronmotor" in str(df.iloc[0,2]) or "Asynschronmotor" in str(df.iloc[0,2])] #Stufe 1
motor.remove("BG 010")
geber = [df.iloc[0,4][-6:]  for df in all_dfs if "Geber" in str(df.iloc[0,2])] #Stufe 2, Kind von Motor
# Anschlussstecker_Klemmkasten = [df.iloc[0,4][-6:] for df in all_dfs if "Klemm" in df.iloc[0,2] or "Anschluss" in df.iloc[0,2]] #Stufe 2, Kind von Motor
flansch_fussgehaeuse = [df.iloc[0,4][-6:]  for df in all_dfs if "flansch" in df.iloc[0,2].lower() or "Fußgehäuse" in df.iloc[0,2]] #Stufe 2, Kind von Getriebe
# Bauteile, die alle Varianten haben: S 001 auf Stufe 1, 
varianten = []
for g in getriebe:
    for m in motor:
        variante = [g,m]
        varianten.append(variante)

# print(varianten[0])
# print(dic_df)



def add_bgs (bg, dic, p, s, p_elt, m):
    dic[p] = [bg, s, p_elt, m]
    if bg[1]=='G':
        bg_df=dic_df.get(bg) 
        p_elt = p
        if bg_df is None:
            print(bg)
        for ind, row in bg_df.iterrows():
            if row["Sachnummer"]==bg:
                print("error in ", bg)
                continue
            p = max(dic.keys())+1
            dic = add_bgs(row["Sachnummer"], dic, p, s+1, p_elt, row["Menge "])
    return dic



def insert_node(df_variante, stufe, eltern, pos_eltern):
    for pos, row in df_variante[(df_variante["stufe"]==stufe) & (df_variante["pos_eltern"]==pos_eltern)].iterrows():
        nextnode = eltern.add_child(name=row["sachnummer"], dist = row["menge"])
        if nextnode.name[1]=='G':
            insert_node(df_variante, stufe+1, nextnode, pos)

def get_tree_format_string (variante):
    dic={}
    dic[1] = ['S 001', 1, 0, 1]
    zwi_df = add_bgs(variante[0], dic, 2, 1, 0, 1)
    final_df = add_bgs(variante[1], zwi_df, len(zwi_df)+1, 1, 0, 1)
    df_variante = pd.DataFrame.from_dict(final_df, orient = "index", columns=["sachnummer","stufe","pos_eltern", "menge"])

    t = Tree(format=1)
    r = t.add_child(name='')
    insert_node(df_variante, 1, r, 0) 
    leafs = df_variante[(df_variante["stufe"]>2)]
    nodes = df_variante[(df_variante["stufe"]<=2)]

    s=t.get_ascii(show_internal=True)
    # print(s)
    return t.write(format = 1)

formats = []
# get_tree_format_string(['BG 024', 'BG 010'])
for v in varianten:
    try:
        formats.append(get_tree_format_string(v))
    except:
        print("not able to format ", v)

print("successfully formated {} products".format(len(formats)))


def encode_variante (format_string):

    enc1 = None
    enc2 = None
    enc3 = None
    enc4 = None
    enc5 = None
    enc6 = None

    if "BG 006" in format_string:
        enc1="FU"
    elif "BG 032" in format_string:
        enc1="FL"
    
    for i in range(1,5):
        if "BG 00{}".format(i) in format_string:
            enc2=str(i)
    
    if "BT 001" in format_string:
        enc3="14"
    elif "BT 002" in format_string:
        enc3="22"

    if "BT 009" in format_string:
        enc4="S"
    elif "BT 017" in format_string:
        enc4="AS"
    
    if "BT 012" in format_string:
        enc5="A"
    elif "BT 014" in format_string:
        enc5="KS"
    elif "BT 016" in format_string:
        enc5="KF"

    if "BT 003" in format_string:
        enc6='1'
    elif "BT 004" in format_string:
        enc6='2'
    elif "BT 018" in format_string:
        enc6='3'
    else:
        enc6='0'
    
    try:
        return enc1+enc2+enc3+enc4+enc5+enc6
    except:
        print("not able to encode {}".format(format_string))
        print(enc1, enc2, enc3, enc4, enc5, enc6)
        return None

# print(encode_variante(formats[10]))  
print(formats[0], "\n", formats[1])
encodings = []
for f in formats:
    encodings.append(encode_variante(f))

# print(encodings)