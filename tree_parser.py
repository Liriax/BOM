from ete3 import Tree
import openpyxl
import pandas
import os


def extract_sequences(df, lst):
    if len(df.columns)<17:
        print("error reading excel file")
        return
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
        # print(sachnummer_lst)
        # print(seq)
        lst.append((sachnummer_lst, seq))
        
            
def cosine_similarity(fm1, fm2):
    dotprod = 0
    betrag_x = 0
    betrag_y = 0
    for i in range(0,len(fm1)):
        x=int(fm1[i])
        y=int(fm2[i])
        dotprod+=x*y
        betrag_x+=x*x
        betrag_y+=y*y
    return dotprod/((betrag_x*betrag_y)**(1/2.0))

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
        test_df_L1=pandas.read_excel(os.path.join(subdir,files[0]))
        test_df_L2=pandas.read_excel(os.path.join(subdir,files[1]))
        test_df_L3=pandas.read_excel(os.path.join(subdir,files[2]))

        test_df = pandas.concat([test_df_L1.iloc[:,:7], test_df_L2.iloc[:,:7], test_df_L3.iloc[:,:7]], axis=0, ignore_index=True)
        nodes = pandas.DataFrame(columns=['sachnummer', 'position', 'pos_eltern', 'stufe', 'menge'])

        position = 0
        last_position=-1

        for i in test_df.index:
            sachnummer = str(test_df.iloc[i,5])
            menge = str(test_df.iloc[i,2])

            if "BG" in sachnummer:
                sachnummer = "BG "+sachnummer[-3:]
                # stufe = int(test_df['Unnamed: 1'][i])

                # elternbaugruppe = sachnummer
                # node_is_new = True
                # for j in nodes.index:
                #     if sachnummer == nodes['sachnummer'][j] and stufe == nodes['stufe'][j]:
                #         node_is_new = False

                # if node_is_new:
                #     position+=1
                #     if stufe==1:
                #         pos_eltern=0
                #     else:
                #         pos_eltern=last_position
                #     new_node = {'sachnummer': sachnummer, 'position': position, 'pos_eltern': pos_eltern,'stufe': stufe, 'menge': menge}
                #     nodes = nodes.append(new_node, ignore_index=True)
                #     last_position=position


            elif 'BT' in sachnummer or ('S' in sachnummer and sachnummer != 'Sachnummer'):
                sachnummer = "BT "+sachnummer[-3:] if "BT" in sachnummer else "S "+sachnummer[-3:]

                # stufe = int(test_df.iloc[i,1])

                formenschluessel = str(test_df.iloc[i,6])
                # pos_eltern=last_position
                if formenschluessel is not None:
                    sachn_formenschl[sachnummer]=formenschluessel

                # leaf_is_new = True
                # for j in nodes.index:
                #     if sachnummer == nodes['sachnummer'][j]  \
                #             and pos_eltern == nodes['pos_eltern'][j]:
                #         leaf_is_new = False

                # if leaf_is_new:
                #     position+=1
                #     new_leaf = {'sachnummer': sachnummer, 'position':position, 'pos_eltern': pos_eltern, 'stufe': stufe, 'menge': menge}
                #     nodes = nodes.append(new_leaf, ignore_index=True)
        
        extract_sequences(test_df_L1,seq_L1)
        extract_sequences(test_df_L2,seq_L2)
        extract_sequences(test_df_L3,seq_L3)

print(len(seq_L1))
print(len(seq_L2))
print(len(seq_L3))

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



