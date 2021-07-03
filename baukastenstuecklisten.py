import pandas as pd
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
    return dfs

left_dfs= split_df_by_NaN(df_left)
right_dfs=split_df_by_NaN(df_right)

all_dfs = [df.rename(columns=df.iloc[1]).drop(df.index[1]) for df in left_dfs + right_dfs if df.empty == False]
dic_df = {} 
for df in all_dfs:
    dic_df[df.iloc[0,4][-6:]] = df.drop(df.index[0]).drop(columns=['Bennung'])

getriebesatz = [df.iloc[0,4][-6:] for df in all_dfs if "Getriebesatz" in str(df.iloc[0,2])] #Stufe 2, Kind von Getriebe
getriebe = [df.iloc[0,4][-6:]  for df in all_dfs if "Getriebe " in df.iloc[0,2]] #Stufe 1
motor = [df.iloc[0,4][-6:] for df in all_dfs if "motor" in str(df.iloc[0,2])] #Stufe 1
geber = [df.iloc[0,4][-6:]  for df in all_dfs if "Geber" in str(df.iloc[0,2])] #Stufe 2, Kind von Motor
Anschlussstecker_Klemmkasten = [df.iloc[0,4][-6:] for df in all_dfs if "Klemm" in df.iloc[0,2] or "Anschluss" in df.iloc[0,2]] #Stufe 2, Kind von Motor
flansch_fussgehaeuse = [df.iloc[0,4][-6:]  for df in all_dfs if "flansch" in df.iloc[0,2].lower() or "Fußgehäuse" in df.iloc[0,2]] #Stufe 2, Kind von Getriebe
# Bauteile, die alle Varianten haben: S 001 auf Stufe 1, 
varianten = []
for g in getriebe:
    for m in motor:
        variante = [g,m]
        varianten.append(variante)

print(varianten[0])
print(len(varianten))
print(dic_df.get('BG 001'))

def df_variante (variante):
    getriebe = dic_df.get(variante[0])    
    motor = dic_df.get(variante[1])
    dic = {}
    dic['S 001'] = [1, 1, None, 1]
    dic[variante[0]] = [2, 1, None, 1]
    dic[variante[1]] = [3, 1, None, 1]
    p = 4
    for ind, row in getriebe.iterrows():
        dic[row["Sachnummer"]] = [p, 2, dic[variante[0]][0],row["Menge "]]
    df = pd.DataFrame.from_dict(dic, orient = 'index', columns=["Position","stufe","pos_elternbaugruppe","menge"])

    return df

print(df_variante(varianten[0]))
