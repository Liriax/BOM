from ete3 import Tree, NodeStyle, TreeStyle, PhyloTree, PhyloNode
import openpyxl
import pandas
import os


def extract_sequences(df, dict):
    for i, row in df.iterrows():
        seq = row[0]
        if isinstance(seq, int) and isinstance(row[1], str):
            seq = int(seq)
            dict[seq]=row[1]
            
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

rootdir = 'datenRF4/AFL'
sachn_formenschl = {}
seq_L1={}
seq_L2={}
seq_L3={}

for subdir, dirs, files in os.walk(rootdir):
    if len(files)>0:
        test_df_L1=pandas.read_excel(os.path.join(subdir,files[0]))
        test_df_L2=pandas.read_excel(os.path.join(subdir,files[1]))
        test_df_L3=pandas.read_excel(os.path.join(subdir,files[2]))

        test_df = pandas.concat([test_df_L1.iloc[:,:7], test_df_L2.iloc[:,:7], test_df_L3.iloc[:,:7]], axis=0, ignore_index=True)
        # print(test_df)
        variante_name = test_df.iloc[0,2]
        print(variante_name)
        nodes = pandas.DataFrame(columns=['sachnummer', 'position', 'pos_eltern', 'stufe', 'menge'])

        position = 0
        last_position=-1

        for i in test_df.index:
            sachnummer = str(test_df['Unnamed: 5'][i])
            menge = str(test_df['Unnamed: 2'][i])

            if "BG" in sachnummer:
                sachnummer = "BG "+sachnummer[-3:]
                stufe = int(test_df['Unnamed: 1'][i])

                elternbaugruppe = sachnummer
                node_is_new = True
                for j in nodes.index:
                    if sachnummer == nodes['sachnummer'][j] and stufe == nodes['stufe'][j]:
                        node_is_new = False

                if node_is_new:
                    position+=1
                    if stufe==1:
                        pos_eltern=0
                    else:
                        pos_eltern=last_position
                    new_node = {'sachnummer': sachnummer, 'position': position, 'pos_eltern': pos_eltern,'stufe': stufe, 'menge': menge}
                    nodes = nodes.append(new_node, ignore_index=True)
                    last_position=position


            elif 'BT' in sachnummer or ('S' in sachnummer and sachnummer != 'Sachnummer'):
                sachnummer = "BT "+sachnummer[-3:] if "BT" in sachnummer else "S "+sachnummer[-3:]

                stufe = int(test_df['Unnamed: 1'][i])

                formenschluessel = str(test_df['Unnamed: 6'][i])
                pos_eltern=last_position
                if formenschluessel is not None:
                    sachn_formenschl[sachnummer]=formenschluessel

                leaf_is_new = True
                for j in nodes.index:
                    if sachnummer == nodes['sachnummer'][j]  \
                            and pos_eltern == nodes['pos_eltern'][j]:
                        leaf_is_new = False

                if leaf_is_new:
                    position+=1
                    new_leaf = {'sachnummer': sachnummer, 'position':position, 'pos_eltern': pos_eltern, 'stufe': stufe, 'menge': menge}
                    nodes = nodes.append(new_leaf, ignore_index=True)
        
        nan_value = float('NaN')
        df_prozess = test_df_L1.iloc[3:,7:]

        extract_sequences(test_df_L1.iloc[3:,7:],seq_L1)
        extract_sequences(test_df_L2.iloc[3:,7:],seq_L2)
        extract_sequences(test_df_L3.iloc[3:,7:],seq_L3)

    
# print(seq_L1)
# print(seq_L2)
# print(seq_L3)
# print(sachn_formenschl)

sim_matrix=[]
for bt in sachn_formenschl.keys():
    lst=[]
    fs = sachn_formenschl.get(bt)
    for bt2 in sachn_formenschl.keys():
        fs2 = sachn_formenschl.get(bt2)
        lst.append(cosine_similarity(fs,fs2))
    sim_matrix.append(lst)
# print(sim_matrix)












nan_value = float('NaN')
nodes.replace("nan", nan_value, inplace=True)
nodes.dropna(how='any', axis=0, inplace=True)
nodes.reset_index(inplace=True)



# def insert_nodes(treenode, currentstufe, index):
#     while index < nodes_maxindex:
#         if float(nodes['stufe'][index]) == float(currentstufe):
#             nextnode = treenode.add_child(name=nodes['sachnummer'][index],dist=nodes['menge'][index])
#             index = index + 1
#             if(index == nodes_maxindex):
#                 return index
#             if float(nodes['stufe'][index]) < float(currentstufe):
#                return index
#             elif float(nodes['stufe'][index]) > float(currentstufe):
#                index = insert_nodes(nextnode, currentstufe + 1, index)


# def insert_leafs(t):
    # for node in t.traverse():
    #     if node.name[0:2] == 'BG':
    #         sachnummer = node.name
    #         pos_eltern = nodes['position'][int(node.name[2:])]
    #         for index in leafs.index:
    #             if leafs['elternbaugruppe'][index] == sachnummer and leafs['pos_eltern'][index] == pos_eltern:
    #                 node.add_child(name=leafs['sachnummer'][index], dist=leafs['menge'][index])



t = Tree(format=1)
r = t.add_child(name='')

def insert_node(df_variante, stufe, eltern, pos_eltern):
    for pos, row in df_variante[(df_variante["stufe"]==stufe) & (df_variante["pos_eltern"]==pos_eltern)].iterrows():
        nextnode = eltern.add_child(name=row["sachnummer"], dist = row["menge"])
        if nextnode.name[1]=='G':
            insert_node(df_variante, stufe+1, nextnode, row['position'])


# insert_node(nodes, 1, r, 0) 

# s=t.get_ascii(show_internal=True)
# print(s)


# pt = PhyloTree(t.write())

# print(pt.write(format = 1))

