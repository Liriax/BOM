from ete3 import Tree
import openpyxl
import pandas
import os

'''this programm extracts the processes' sequence number for each corresponding Baugruppe and Bauteile, for each station'''
'''it also extracts all the Bauteil - Formenschlüssel pairs'''

def extract_sequences(df, lst):
    is_NaN = df.iloc[:,:7].isnull()
    row_all_NaN = is_NaN.all(axis=1)
    NaN_rows = df[row_all_NaN].index.tolist()
    dfs = []
    dfs.append(df.iloc[0:NaN_rows[0],:])
    for n in range(0, len(NaN_rows)-1):
        start = NaN_rows[n]+1
        end = NaN_rows[n+1]
        dfs.append(df.iloc[start:end, :])
    dfs.append(df.iloc[NaN_rows[len(NaN_rows)-1]+1:,:])
    dats = [x for x in dfs if x.empty==False]
    for d in dats:
        sachnummer_lst = [x for x in d.iloc[:,5] if x is not None]
        seq = [x for x in d.iloc[:,7][d.iloc[:,7].notnull()]]
        lst.append((sachnummer_lst, seq))
        

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


rootdir = 'datenRF4'
sachn_formenschl = {}
seq_L1=[]
seq_L2=[]
seq_L3=[]

for subdir, dirs, files in os.walk(rootdir):
    if len(files)>0:
        test_df_L1=pandas.read_excel(os.path.join(subdir,files[0]),header=None,index_col=None)
        test_df_L2=pandas.read_excel(os.path.join(subdir,files[1]),header=None,index_col=None)
        test_df_L3=pandas.read_excel(os.path.join(subdir,files[2]),header=None,index_col=None)

        test_df = pandas.concat([test_df_L1.iloc[:,:7], test_df_L2.iloc[:,:7], test_df_L3.iloc[:,:7]], axis=0, ignore_index=True)
        nodes = pandas.DataFrame(columns=['sachnummer', 'position', 'pos_eltern', 'stufe', 'menge'])

        position = 0
        last_position=-1

        for i in test_df.index:
            sachnummer = str(test_df.iloc[i,5])
            menge = str(test_df.iloc[i,2])

            if "BG" in sachnummer:
                sachnummer = "BG "+sachnummer[-3:]

            elif 'BT' in sachnummer or ('S' in sachnummer and sachnummer != 'Sachnummer'):
                sachnummer = "BT "+sachnummer[-3:] if "BT" in sachnummer else "S "+sachnummer[-3:]
                formenschluessel = str(test_df.iloc[i,6])
                if formenschluessel is not None:
                    sachn_formenschl[sachnummer]=formenschluessel
   
        if len(test_df_L1.columns)>=17:
            extract_sequences(test_df_L1,seq_L1)
            
        if len(test_df_L2.columns)>=17:
            extract_sequences(test_df_L2,seq_L2)
           
        if len(test_df_L3.columns)>=17:
            extract_sequences(test_df_L3,seq_L3)


sachn_formenschl.pop("BT T18", None)




def find_process (bt_lst):
    stations = [seq_L1,seq_L2,seq_L3]
    ret = []
    for i in range(0,3):
        seqs=[]
        station = stations[i]
        for tup in station:
            components=tup[0]
            sequence = tup[1]
            if set(bt_lst) <= set(components):
                for x in sequence:
                    seqs.append(x)
        ret.append(list(set(seqs)))
    return ret


# print(find_process(["BG 008"]))



