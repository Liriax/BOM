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

getriebesatz = [(df.iloc[0,4][-3:], df.drop(df.index[0])) for df in all_dfs if "Getriebesatz" in str(df.iloc[0,2])]
getriebe = [(df.iloc[0,4][-3:], df.drop(df.index[0]))  for df in all_dfs if "Getriebe " in df.iloc[0,2]]
motor = [(df.iloc[0,4][-3:], df.drop(df.index[0])) for df in all_dfs if "motor" in str(df.iloc[0,2])]
geber = [(df.iloc[0,4][-3:], df.drop(df.index[0]))  for df in all_dfs if "Geber" in str(df.iloc[0,2])]
Anschlussstecker_Klemmkasten = [df for df in all_dfs if "Klemm" in df.iloc[0,2] or "Anschluss" in df.iloc[0,2]]
flansch_fussgehaeuse = [(df.iloc[0,4][-3:], df.drop(df.index[0]))  for df in all_dfs if "flansch" in df.iloc[0,2].lower() or "Fußgehäuse" in df.iloc[0,2]]
print(geber[0])
