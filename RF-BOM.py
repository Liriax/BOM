from ete3 import Tree, NodeStyle, TreeStyle
import re
from pulp import *
import numpy as np

t1 = Tree('(((1:4,2:4),3:1), (7:1,(4:3,5:1,6:1)));')
t2 = Tree('(((1:4,2:4),(3:2, 9:1)), ((8:1, 5:1, 6:1), 7:1));')
print (t1, t2)

''' returns a list of tuples (node, hierarchy level)
 input t1=root, lst=[(t1,0)], level=0 '''
def get_nodes_with_hierachy_level(t1, lst, level):
    if t1.is_leaf():
        return lst
    for child in t1.children:
        lst.append((child, level+1))
        lst = get_nodes_with_hierachy_level(child, lst, level+1)
    return lst


def generate_matrixes(t1, t2):
    t1_all_nodes = get_nodes_with_hierachy_level(t1,[(t1,0)],0)
    t1_dic = {}
    
    t2_all_nodes = get_nodes_with_hierachy_level(t2,[(t2,0)],0)
    t2_dic = {}
   
    t1_nodes = [x for x in t1_all_nodes if x[0].is_leaf()==False]
    t2_nodes = [x for x in t2_all_nodes if x[0].is_leaf()==False]
    '''fill the dictionaries:
    the dictionary will be have TreeNode objects of inner nodes as keys and sub-dictionaries as value
    sub-dictionaries will habe leaf names as keys and a tuple (relative hierarchy level, quantity of the leaf) as value
    relative hierarchy level = hierarchy level of the leaf - hierarchy level of the inner node'''
    for node in t1_nodes:
        node_dic={}
        for x in get_nodes_with_hierachy_level(node[0],[],node[1]):
            if x[0].is_leaf():
                node_dic[x[0].name]=(x[1]-node[1],x[0].dist) # (relative hierarchy level, quantity)
        t1_dic[node[0]]=node_dic

    for node in t2_nodes:
        node_dic={}
        for x in get_nodes_with_hierachy_level(node[0],[],node[1]):
            if x[0].is_leaf():
                node_dic[x[0].name]=(x[1]-node[1],x[0].dist) # (relative hierarchy level, quantity)
        t2_dic[node[0]]=node_dic

    t1_comp = [x[0].name for x in t1_all_nodes if x[0].is_leaf()==True]
    t2_comp = [x[0].name for x in t2_all_nodes if x[0].is_leaf()==True]
    all_comp = list(set(t1_comp)|set(t2_comp))
    all_comp.sort()

    '''fill the matrixes: the matrixes are dictionaries with inner node id as key and 
    a list of components as value (just like the matrixes on top left, page 8 of the paper)'''
    mat1={}
    for n in range(0, len(t1_nodes)):
        mat1[n+1]=[1/t1_dic.get(t1_nodes[n][0]).get(x)[0]*t1_dic.get(t1_nodes[n][0]).get(x)[1] if x in [x.name for x in t1_nodes[n][0].search_nodes()]  else 0 for x in all_comp]
    mat2={}
    for n in range(0, len(t2_nodes)):
        mat2[n+1]=[1/t2_dic.get(t2_nodes[n][0]).get(x)[0]*t2_dic.get(t2_nodes[n][0]).get(x)[1] if x in [x.name for x in t2_nodes[n][0].search_nodes()]  else 0 for x in all_comp]

    return mat1, mat2, t1_nodes, t2_nodes


def find_same_nodes(mat1, mat2):
    same_nodes_t1=[]
    same_nodes_t2=[]
    common_node_values=[x for x in list(mat1.values()) if x in list(mat2.values())]
    for common_node_value in common_node_values:
        common_node_t1=list(mat1.keys())[list(mat1.values()).index(common_node_value)]
        common_node_t2=list(mat2.keys())[list(mat2.values()).index(common_node_value)]
        same_nodes_t1.append(nodes1[common_node_t1-1])
        same_nodes_t2.append(nodes2[common_node_t2-1])
    return same_nodes_t1, same_nodes_t2

# {1: [1.3333333333333333, 1.3333333333333333, 0.5, 1.0, 0.3333333333333333, 0.3333333333333333, 0.5, 0, 0], 2: [2.0, 2.0, 1.0, 0, 0, 0, 0, 0, 0], 3: [4.0, 4.0, 0, 0, 0, 0, 0, 0, 0], 4: [0, 0, 0, 1.5, 0.5, 0.5, 1.0, 0, 0], 5: [0, 0, 0, 3.0, 1.0, 1.0, 0, 0, 0]}
# {1: [1.3333333333333333, 1.3333333333333333, 0.6666666666666666, 0, 0.3333333333333333, 0.3333333333333333, 0.5, 0.3333333333333333, 0.3333333333333333], 2: [2.0, 2.0, 1.0, 0, 0, 0, 0, 0, 0.5], 3: [4.0, 4.0, 0, 0, 0, 0, 0, 0, 0], 4: [0, 0, 2.0, 0, 0, 0, 0, 0, 1.0], 5: [0, 0, 0, 0, 0.5, 0.5, 1.0, 0.5, 0], 6: [0, 0, 0, 0, 1.0, 1.0, 0, 1.0, 0]}

''' returns (d1, d2), where d1 is the distance between node 1 and node 2 while mat1 is mapped to mat2 and d2 is the distance between node 1 and node 2 while mat2 is mapped to mat1'''
def cal_distance (mat1, mat2, node1, node2):
    n_components=len(mat1[node1])
    l1 = [mat1[node1][x]-mat2[node2][x] for x in range (0, n_components)]
    l2 = [mat2[node2][x]-mat1[node1][x] for x in range (0, n_components)]
    return round(sum([d for d in l1 if d>=0]),2), round(sum([d for d in l2 if d>=0]),2)

def create_dist_mapping_matrix (mat1, mat2):
    n_nodes_mat1 = len(mat1)
    n_nodes_mat2 = len(mat2)
    T12 = [[cal_distance(mat1, mat2, i, j)[0] for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
    T21 = [[cal_distance(mat1, mat2, j, i)[1] for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
    return T12, T21

'''the second step is to implement the integer programming model:
    input = the two matrixes obtained from generate_matrixes(t1, t2)
    output:
    ip1 = matrix that matches the nodes of t1 to nodes of t2, producing the smallest total distance
    ip2 = matrix that matches the nodes of t2 to nodes of t1, producing the smallest total distance'''  
def integer_programming (mat1, mat2):
    n_nodes_mat1 = len(mat1)
    n_nodes_mat2 = len(mat2)

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("The Node Matching Problem", LpMinimize)

    # The problems variables
    t1 = [[LpVariable("z{}{}".format(i, j), 0, 1, LpInteger) for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
    t2 = [[LpVariable("w{}{}".format(i, j), 0, 1, LpInteger) for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
    

    dm1, dm2 = create_dist_mapping_matrix(mat1, mat2)
    RF_BOM_max =sum([sum(mat1[i]) for i in range(1, n_nodes_mat1+1)])+sum([sum(mat2[i]) for i in range(1, n_nodes_mat2+1)])

    # The objective function is added to 'prob' first
    prob += sum([sum([t1[i][j]*dm1[i][j] for j in range(0, n_nodes_mat2)]) for i in range(0, n_nodes_mat1)])+ \
            sum([sum([t2[i][j]*dm2[i][j] for j in range(0, n_nodes_mat1)]) for i in range(0, n_nodes_mat2)]), "Total distance between matched nodes"

    # The constraints are entered
    constraints_t1 = [sum(t1[i]) == 1 for i in range(0, n_nodes_mat1)]
    constraints_t2 = [sum(t2[i]) == 1 for i in range(0, n_nodes_mat2)]

    for cst in constraints_t1:
        prob += cst
    for cst in constraints_t2:
        prob += cst

    # prob.writeLP("NodeMatching.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    RF_BOM = value(prob.objective)
    # The optimised objective function value is printed to the screen
    print("Total node matching distance = ", value(prob.objective))

    RF_BOM_norm = (RF_BOM_max-RF_BOM)/RF_BOM_max
    return RF_BOM_norm


# Basic tree style
ts = TreeStyle()
ts.show_leaf_name = True
ts.show_branch_length = True

# Creates an independent node style for each node, which is
# initialized with a red foreground color.
for n in t1.traverse():
   nstyle = NodeStyle()
   nstyle["fgcolor"] = "red"
   nstyle["size"] = 15
   n.set_style(nstyle)
for n in t2.traverse():
   nstyle = NodeStyle()
   nstyle["fgcolor"] = "red"
   nstyle["size"] = 15
   n.set_style(nstyle)

mat1, mat2, nodes1, nodes2 = generate_matrixes(t1, t2)

# print(mat1)
# print(mat2)
# print(cal_distance(mat1, mat2, 4,5))
# dm1, dm2 = create_dist_mapping_matrix(mat1, mat2)
# for l in dm1:
#     print(l)
#     print("")
# for l in dm2:
#     print(l)
#     print("")

cm1, cm2 = find_same_nodes(mat1, mat2)

#plot the common nodes green
for l in [cm1, cm2]:
    for node in l:
        node[0].img_style["size"] = 20
        node[0].img_style["fgcolor"] = "green"
        for child in node[0].search_nodes():
            child.img_style["size"] = 20
            child.img_style["fgcolor"] = "green"


# t1.show(tree_style=ts) # requires PyQt5
# t2.show(tree_style=ts) # requires PyQt5




print("normalized RF distance is ", integer_programming(mat1, mat2))
