from ete3 import Tree, NodeStyle, TreeStyle, PhyloTree, PhyloNode
import openpyxl
import pandas

test_df_L1 = pandas.read_excel('datenRF4/AFL/i=23,28, Fremdlüfter, Abtriebswelle=14/Prozesse_Stückliste_L1_S4_i23,28_Fremd.xlsx')
test_df_L2 = pandas.read_excel('datenRF4/AFL/i=23,28, Fremdlüfter, Abtriebswelle=14/Prozesse_Stückliste_L2_S4_i23_Fremdlüfter.xlsx')
test_df_L3 = pandas.read_excel('datenRF4/AFL/i=23,28, Fremdlüfter, Abtriebswelle=14/Prozesse_Stückliste_L3_S4_i23_fremdlüfter_abtriebswelle14.xlsx')

test_df = pandas.concat([test_df_L1, test_df_L2, test_df_L3], ignore_index=True)



elternbaugruppe = 'none'

nodes = pandas.DataFrame(columns=['sachnummer', 'position', 'stufe', 'menge'])
leafs = pandas.DataFrame(columns=['sachnummer', 'elternbaugruppe', 'pos_elternbaugruppe', 'menge'])
for i in test_df.index:
    sachnummer = str(test_df['Unnamed: 5'][i])
    if "BG" in sachnummer:

        elternbaugruppe = sachnummer
        position = str(test_df['Produktinformationen'][i])
        pos_elternbaugruppe = position
        stufe = str(test_df['Unnamed: 1'][i])
        menge = str(test_df['Unnamed: 2'][i])

        node_is_new = True
        for j in nodes.index:
            if sachnummer == nodes['sachnummer'][j] and position == nodes['position'][j]:
                node_is_new = False

        if node_is_new:
            new_node = {'sachnummer': sachnummer, 'position': position, 'stufe': stufe, 'menge': menge}
            nodes = nodes.append(new_node, ignore_index=True)


    elif 'BT' in sachnummer or ('S' in sachnummer and sachnummer != 'Sachnummer'):

        menge = str(test_df['Unnamed: 2'][i])

        leaf_is_new = True
        for j in leafs.index:
            if sachnummer == leafs['sachnummer'][j] and elternbaugruppe == leafs['elternbaugruppe'][j] \
                    and pos_elternbaugruppe == leafs['pos_elternbaugruppe'][j]:
                leaf_is_new = False

        if leaf_is_new:

            new_leaf = {'sachnummer': sachnummer, 'elternbaugruppe': elternbaugruppe,
                        'pos_elternbaugruppe': pos_elternbaugruppe, 'menge': menge}
            leafs = leafs.append(new_leaf, ignore_index=True)

t = Tree()


nodes.sort_values(by=['position'])
nan_value = float('NaN')
nodes.replace("nan", nan_value, inplace=True)
nodes.dropna(how='any', axis=0, inplace=True)
nodes.reset_index(inplace=True)

leafs.replace("nan", nan_value, inplace=True)
leafs.dropna(how='any', axis=0, inplace=True)
leafs.reset_index(inplace=True)


nodes_maxindex = nodes['position'].size
leafs_maxindex = leafs['menge'].size

print(nodes)
print(leafs)

def insert_nodes(treenode, currentstufe, index):
    while index < nodes_maxindex:
        if float(nodes['stufe'][index]) == float(currentstufe):
            nextnode = treenode.add_child(name='BG' + str(index))
            index = index + 1
            if(index == nodes_maxindex):
                return index
            if float(nodes['stufe'][index]) < float(currentstufe):
               return index
            elif float(nodes['stufe'][index]) > float(currentstufe):
               index = insert_nodes(nextnode, currentstufe + 1, index)


def insert_leafs(t):
    for node in t.traverse():
        if node.name[0:2] == 'BG':
            sachnummer = nodes['sachnummer'][int(node.name[2:])]
            pos_elternbaugruppe = nodes['position'][int(node.name[2:])]
            for index in leafs.index:
                if leafs['elternbaugruppe'][index] == sachnummer and leafs['pos_elternbaugruppe'][index] == pos_elternbaugruppe:
                    node.add_child(name='BT' + str(index))

r = t.add_child(name='root')
insert_nodes(r, 1, 0)
insert_leafs(t)
print(t)

print(t.write)

pt = PhyloTree(t.write())

print(pt)

