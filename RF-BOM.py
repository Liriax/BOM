from ete3 import Tree
import re
t1 = Tree('(((a,b),c), ((e, f), g));')
t2 = Tree('(((a,c),b), ((e, f), g));')
print([x for x in t1.search_nodes() if x.is_leaf()==False])
# rf_lst = t1.robinson_foulds(t2)
# rf = rf_lst[0]
# max_rf = rf_lst[1]
# common_leaves = rf_lst[2]
# parts_t1 = rf_lst[3]
# parts_t2 = rf_lst[4]

print (t1, t2)
# print ("RF distance is %s over a total of %s" %(rf, max_rf))
# print ("Partitions in tree2 that were not found in tree1:", parts_t2 - parts_t1)
# print ("Partitions in tree1 that were not found in tree2:", parts_t1 - parts_t2)
# print ("common parts: ", parts_t1)


def generate_matrixes(t1, t2):
#     t1_com = [x for x in re.split(' ', re.sub('[(),]',' ', t1)) if x!='']
#     t2_com = [x for x in re.split(' ', re.sub('[(),]',' ', t2)) if x!='']
#     all_comp = list(set(t1_com)|set(t2_com))
#     mat1={}
#     mat2={}
#     n=1
#     while t1!='':
#         mat1[n]=re.split(',\s*(?![^()]*\))',re.findall('(?<=\().*(?=\))',t2)[0])
#         n+=1
    t1_nodes = [x for x in t1.search_nodes() if x.is_leaf()==False]
    t2_nodes = [x for x in t2.search_nodes() if x.is_leaf()==False]
    t1_comp = [x for x in t1.search_nodes() if x.is_leaf()==True]
    t2_comp = [x for x in t2.search_nodes() if x.is_leaf()==True]
    all_comp = list(set(t1_comp)|set(t2_comp))
    mat1={}
    for n in range(0, len(t1_nodes)):
        mat1[n+1]=[x if x in t1_comp  else 0 for x in all_comp]
    mat2={}
    for n in range(0, len(t2_nodes)):
        mat2[n+1]=[x if x in t2_comp  else 0 for x in all_comp]
    return mat1, mat2

        


mat1, mat2 = generate_matrixes(t1, t2)
print(mat1)
# t1='(((a,b),c),  ((e, f), g))'
# t2='(((a,c),d),((e, f), g))'
# print(re.split(',\s*(?![^()]*\))',re.findall('(?<=\().*(?=\))',t2)[0]))


