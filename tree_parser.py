from ete3 import Tree, NodeStyle, TreeStyle
import openpyxl
import pandas

test_df_L1 = pandas.read_excel('datenRF4/AFL/i=23,28, Fremdlüfter, Abtriebswelle=14/Prozesse_Stückliste_L1_S4_i23,28_Fremd.xlsx')
test_df_L2 = pandas.read_excel('datenRF4/AFL/i=23,28, Fremdlüfter, Abtriebswelle=14/Prozesse_Stückliste_L2_S4_i23_Fremdlüfter.xlsx')
test_df_L3 = pandas.read_excel('datenRF4/AFL/i=23,28, Fremdlüfter, Abtriebswelle=14/Prozesse_Stückliste_L3_S4_i23_fremdlüfter_abtriebswelle14.xlsx')

test_df = pandas.concat([test_df_L1, test_df_L2, test_df_L3], ignore_index=True)



elternbaugruppe = 'none'

nodes = pandas.DataFrame(columns=['sachnummer', 'position', 'stufe', 'menge'])
leafs = pandas.DataFrame(columns=['sachnummer', 'elternbaugruppe', 'menge'])
for i in test_df.index:
    sachnummer = str(test_df['Unnamed: 5'][i])
    if "BG" in sachnummer:

        elternbaugruppe = sachnummer
        position = str(test_df['Produktinformationen'][i])
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
            if sachnummer == leafs['sachnummer'][j] and elternbaugruppe == leafs['elternbaugruppe'][j]:
                leaf_is_new = False

        if leaf_is_new:

            new_leaf = {'sachnummer': sachnummer, 'elternbaugruppe': elternbaugruppe, 'menge': menge}
            leafs = leafs.append(new_leaf, ignore_index=True)


print(nodes)
print(leafs)

test_tree = Tree()

first_level_nodes = {}
second_level_nodes = {}

#for i in nodes.index:
#    if int(nodes['stufe'][i]) == 1:
#        print('che guevara')
#        first_level_nodes[nodes['sachnummer'][j], nodes['position']]

#for x in first_level_nodes:
#    test_tree.add_child(name=x['sachnummer'])
##test_tree.add_child(name=row[1])
#print(test_tree)