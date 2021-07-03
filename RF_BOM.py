import PyQt5
from ete3 import Tree #, NodeStyle, TreeStyle
import re
from pulp import *
import numpy as np

'''recursively returns a list of tuples (node, hierarchy level)
input t1=root, lst=[(t1,0)], level=0 '''
def get_nodes_with_hierachy_level(t1, lst, level):
    if t1.is_leaf():
        return lst
    for child in t1.children:
        lst.append((child, level+1))
        lst = get_nodes_with_hierachy_level(child, lst, level+1)
    return lst


class TreePair:

    '''initialize a TreePair object that consists of 2 trees and generate their matrixes for further comparison'''
    def __init__(self, t1, t2):
        self.t1=t1
        self.t2=t2
        t1_all_nodes = get_nodes_with_hierachy_level(t1,[(t1,0)],0)
        
        t2_all_nodes = get_nodes_with_hierachy_level(t2,[(t2,0)],0)
        t1_comp = [x[0].name for x in t1_all_nodes if x[0].is_leaf()==True]
        t2_comp = [x[0].name for x in t2_all_nodes if x[0].is_leaf()==True]
        all_comp = list(set(t1_comp)|set(t2_comp))
        all_comp.sort()

        self.all_comp = all_comp

        self.t1_inner_nodes = [x for x in t1_all_nodes if x[0].is_leaf()==False]
        self.t2_inner_nodes = [x for x in t2_all_nodes if x[0].is_leaf()==False]

        t1_dic={}
        t2_dic={}
        '''fill the dictionaries:
        the dictionary will be have TreeNode objects of inner nodes as keys and sub-dictionaries as value
        sub-dictionaries will habe leaf names as keys and a tuple (relative hierarchy level, quantity of the leaf) as value
        relative hierarchy level = hierarchy level of the leaf - hierarchy level of the inner node'''
        for node in self.t1_inner_nodes:
            node_dic={}
            for x in get_nodes_with_hierachy_level(node[0],[],node[1]):
                if x[0].is_leaf():
                    rel_hier_level = x[1]-node[1]
                    quantity = x[0].dist
                    quantity *= x[0].up.dist if x[0].up != node[0] else 1
                    node_dic[x[0].name]=(rel_hier_level,quantity) # (relative hierarchy level, quantity)

            t1_dic[node[0]]=node_dic

        print (t1_dic)
        for node in self.t2_inner_nodes:
            node_dic={}
            for x in get_nodes_with_hierachy_level(node[0],[],node[1]):
                if x[0].is_leaf():
                    rel_hier_level = x[1]-node[1]
                    quantity = x[0].dist
                    quantity *= x[0].up.dist if x[0].up != node[0] else 1
                    node_dic[x[0].name]=(rel_hier_level,quantity) # (relative hierarchy level, quantity)
            t2_dic[node[0]]=node_dic
 
        '''generate the tree matrixes of 2 trees'''
        '''Here the function generates 2 matrixes, alternatively we could generate only 1 matrix for 1 tree, but in that case we
        would need the all_comp as an input, that has all the possible components listed
        the matrix would look like this:
        {1: [1.3333333333333333, 1.3333333333333333, 0.5, 1.0, 0.3333333333333333, 0.3333333333333333, 0.5, 0, 0], 2: [2.0, 2.0, 1.0, 0, 0, 0, 0, 0, 0], 3: [4.0, 4.0, 0, 0, 0, 0, 0, 0, 0], 4: [0, 0, 0, 1.5, 0.5, 0.5, 1.0, 0, 0], 5: [0, 0, 0, 3.0, 1.0, 1.0, 0, 0, 0]}
        {1: [1.3333333333333333, 1.3333333333333333, 0.6666666666666666, 0, 0.3333333333333333, 0.3333333333333333, 0.5, 0.3333333333333333, 0.3333333333333333], 2: [2.0, 2.0, 1.0, 0, 0, 0, 0, 0, 0.5], 3: [4.0, 4.0, 0, 0, 0, 0, 0, 0, 0], 4: [0, 0, 2.0, 0, 0, 0, 0, 0, 1.0], 5: [0, 0, 0, 0, 0.5, 0.5, 1.0, 0.5, 0], 6: [0, 0, 0, 0, 1.0, 1.0, 0, 1.0, 0]}'''
    
        '''fill the matrixes: the matrixes are dictionaries with inner node id as key and 
        a list of components as value (just like the matrixes on top left, page 8 of the paper)'''
        
        mat1={}
        for n in range(0, len(self.t1_inner_nodes)):
            mat1[n+1]=[1/t1_dic.get(self.t1_inner_nodes[n][0]).get(x)[0]*t1_dic.get(self.t1_inner_nodes[n][0]).get(x)[1] if x in [x.name for x in self.t1_inner_nodes[n][0].search_nodes()]  else 0 for x in all_comp]
        mat2={}
        for n in range(0, len(self.t2_inner_nodes)):
            mat2[n+1]=[1/t2_dic.get(self.t2_inner_nodes[n][0]).get(x)[0]*t2_dic.get(self.t2_inner_nodes[n][0]).get(x)[1] if x in [x.name for x in self.t2_inner_nodes[n][0].search_nodes()]  else 0 for x in all_comp]

        self.t1_dic = t1_dic
        self.t2_dic = t2_dic
        self.mat1=mat1
        self.mat2=mat2


        

    '''identifies common nodes and subtrees, returns two lists of TreeNode objects'''
    def find_same_nodes(self):
        mat1=self.mat1
        mat2=self.mat2
        nodes1=self.t1_inner_nodes
        nodes2=self.t2_inner_nodes
        same_nodes_t1=[]
        same_nodes_t2=[]
        same_nodes_t1_t2=[]
        common_node_values=[x for x in list(mat1.values()) if x in list(mat2.values())]
        for common_node_value in common_node_values:
            common_node_t1=list(mat1.keys())[list(mat1.values()).index(common_node_value)]
            common_node_t2=list(mat2.keys())[list(mat2.values()).index(common_node_value)]
            same_nodes_t1.append(nodes1[common_node_t1-1][0])
            same_nodes_t2.append(nodes2[common_node_t2-1][0])
            same_nodes_t1_t2.append((nodes1[common_node_t1-1][0], nodes2[common_node_t2-1][0]))
        return same_nodes_t1, same_nodes_t2, same_nodes_t1_t2

    ''' returns (d1, d2), where d1 is the distance between node 1 and node 2 while mat1 is mapped to 
    mat2 and d2 is the distance between node 1 and node 2 while mat2 is mapped to mat1'''
    def cal_distance (self, node1, node2, consider_comp_similarity=False, comp_similarity_matrix = None):
        mat1=self.mat1
        mat2=self.mat2
        t1_inner_nodes = self.t1_inner_nodes
        t2_inner_nodes = self.t2_inner_nodes
        t1_dic=self.t1_dic
        t2_dic=self.t2_dic
        all_comp = self.all_comp
        n_components=len(self.all_comp)

        l1 = [mat1[node1][x]-mat2[node2][x] for x in range (0, n_components)]
        l2 = [mat2[node2][x]-mat1[node1][x] for x in range (0, n_components)]

        dist1, dist2 = sum([d for d in l1 if d>=0]), sum([d for d in l2 if d>=0])
        
        '''if component similarities are to be considered: use another interger programming model to find out
        the optimal component to component matching and substract the total similarities of matched components from 
        the distance between node 1 and node 2 while mat1 is mapped to mat2 only'''
        if consider_comp_similarity:
            #	- t12: (4, 4, 1, 0, 0, 0, 0, 0, 0)
	        #   - t24: (0, 0, 2, 0, 0, 0, 0, 0, 1)
            t1n=[t1_dic.get(t1_inner_nodes[node1-1][0]).get(x)[1] if x in [x.name for x in t1_inner_nodes[node1-1][0].search_nodes()] else 0 for x in all_comp]
            t2n=[t2_dic.get(t2_inner_nodes[node2-1][0]).get(x)[1] if x in [x.name for x in t2_inner_nodes[node2-1][0].search_nodes()] else 0 for x in all_comp]
            ind_t1n_diff_t2n = [x for x in range(0, len(t1n)) if t2n[x]<t1n[x]] # indexes of all components in t1n that are not in t2n
            t1n_minus_t2n_positive = [t1n[i]-t2n[i] if t1n[i]>t2n[i] else 0 for i in range(0, len(t1n))] # list of (t1n - t2n)>=0
            ind_t2n_diff_t1n = [x for x in range(0, len(t2n)) if t2n[x]>t1n[x]]
            t2n_minus_t1n_positive = [t2n[i]-t1n[i] if t1n[i]<t2n[i] else 0 for i in range(0, len(t1n))]
            
            # ind_t1n_diff_t2n: [0, 1]
            # ind_t2n_diff_t1n: [2, 8]
            # t12-t24: (4, 4, 0, 0, 0, 0, 0, 0, 0)
            # t24-t12: (0, 0, 1, 0, 0, 0, 0, 0, 1)
 
            prob = LpProblem("the component matching problem", LpMaximize)
            pvar1 = [[LpVariable("t1.replace{}by{}".format(i+1, j+1), 0, None, LpInteger) for j in ind_t2n_diff_t1n ] for i in ind_t1n_diff_t2n]            

            prob+= sum([sum([pvar1[ind_t1n_diff_t2n.index(i)][ind_t2n_diff_t1n.index(j)]*comp_similarity_matrix[i][j] for j in ind_t2n_diff_t1n])\
            /t1_dic.get(self.t1_inner_nodes[node1-1][0]).get(str(i+1))[0] for i in ind_t1n_diff_t2n])
            
            constraints_t1 = [pvar1[ind_t1n_diff_t2n.index(i)][ind_t2n_diff_t1n.index(j)] <= min(t2n_minus_t1n_positive[j], t1n_minus_t2n_positive[i]) for j in ind_t2n_diff_t1n for i in ind_t1n_diff_t2n]
            
 

            for cst in constraints_t1:
                prob += cst
            

            prob.solve(solver = PULP_CBC_CMD(msg=0))
            
            # print(value(prob.objective))
            dist1 -= value(prob.objective) if value(prob.objective) is not None else 0

        return round(dist1,2), round(dist2,2)

    '''creates distance mapping matrixes as nested lists using cal_distance'''
    def create_dist_mapping_matrix (self, consider_comp_similarity=False, comp_similarity_matrix = None):
        mat1=self.mat1
        mat2=self.mat2
        n_nodes_mat1 = len(mat1)
        n_nodes_mat2 = len(mat2)
        if not consider_comp_similarity:
            T12 = [[self.cal_distance(i, j)[0] for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
            T21 = [[self.cal_distance(j, i)[1] for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        else:
            T12 = [[self.cal_distance(i, j, True, comp_similarity_matrix)[0] for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
            T21 = [[self.cal_distance(j, i, True, comp_similarity_matrix)[1] for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        

        return T12, T21

    '''computes the normalizes rf similarity between two tree matrixes using an interger programming model:
        input = the two matrixes obtained from generate_matrixes(t1, t2)
        output: normalized rf metric, values of constraint variables'''  
    def compute_rf_bom (self, consider_comp_similarity=False, comp_similarity_matrix = None):
        mat1=self.mat1
        mat2=self.mat2
        n_nodes_mat1 = len(mat1)
        n_nodes_mat2 = len(mat2)

        # Create the 'prob' variable to contain the problem data
        prob = LpProblem("The Node Matching Problem", LpMinimize)

        # The problems variables
        t1 = [[LpVariable("z{}{}".format(i, j), 0, 1, LpInteger) for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
        t2 = [[LpVariable("w{}{}".format(i, j), 0, 1, LpInteger) for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        

        if not consider_comp_similarity:
            dm1, dm2 = self.create_dist_mapping_matrix()
        else:
            dm1, dm2 = self.create_dist_mapping_matrix(True, comp_similarity_matrix)
        
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
        prob.solve(solver = PULP_CBC_CMD(msg=0))

        # The status of the solution is printed to the screen
        # print("Status:", LpStatus[prob.status])

        # Each of the variables is printed with it's resolved optimum value
        # for v in prob.variables():
        #     print(v.name, "=", v.varValue)

        node_matching_matrix1 = []
        node_matching_matrix2 = []
        for v in prob.variables():
            if v.name[0]=='z' and v.varValue==1:
                node_matching_matrix1.append((int(v.name[1]), int(v.name[2])))
            if v.name[0]=='w' and v.varValue==1: 
                node_matching_matrix2.append((int(v.name[1]), int(v.name[2])))

        RF_BOM = value(prob.objective)
        # The optimised objective function value is printed to the screen
        # print("Total node matching distance = ", value(prob.objective))

        if RF_BOM is None:
            RF_BOM_norm = -1
        else:
            RF_BOM_norm = (RF_BOM_max-RF_BOM)/RF_BOM_max
        return RF_BOM_norm, node_matching_matrix1, node_matching_matrix2

    '''returns a dictionary with tuples (i, j) as keys and rf-distance between node i of tree 1 and node j of tree 2 as values'''
    def find_sim_nodes(self, consider_comp_similarity=False, comp_similarity_matrix = None):
        t1=self.t1
        t2=self.t2
        mat1=self.mat1
        mat2=self.mat2
        nodes1=self.t1_inner_nodes
        nodes2=self.t2_inner_nodes
        tree_dist, nmm1, nmm2 = self.compute_rf_bom() if not consider_comp_similarity else self.compute_rf_bom(True, comp_similarity_matrix)
        
        mapped_nodes = [x for x in nmm1 if (x[1], x[0]) in nmm2]
        # print(mapped_nodes)
        nodes_rf_boms = {}
        for pair in mapped_nodes:
            n1 = nodes1[pair[0]-1][0]
            n2 = nodes2[pair[1]-1][0]
            tp = TreePair(n1,n2)
            mat_n1=tp.mat1
            mat_n2=tp.mat2
            rf_dist = tp.compute_rf_bom()[0] if not consider_comp_similarity else tp.compute_rf_bom(True, comp_similarity_matrix)[0]
            n1_name = n1.name
            n2_name = n2.name
            if rf_dist>=0:
                nodes_rf_boms[(n1_name, n2_name)]=rf_dist

        return nodes_rf_boms


class TreeCompare:
    def __init__(self, t, trees):
        self.trees = trees
        self.t = t

    def find_same_nodes(self):
        same_nodes = []
        for tree in self.trees:
            tp = TreePair(self.t, tree)
            same_nodes.append(tp.find_same_nodes()[2])
        print(same_nodes)
        return same_nodes
    def find_similar_nodes(self):
        similar_nodes = []
        for tree in self.trees:
            tp = TreePair(self.t, tree)
            sim_nodes = tp.find_sim_nodes()
            similar_nodes.append(sim_nodes)
        return similar_nodes

# all end-components should be encoded with numbers from 1 to n_components

t1 = Tree('(((1:4,2:4)a:2,3:1)b:1, (7:1,(4:3,5:1,6:1)c:1)d:1);', format=1)
t2 = Tree('(((1:4,2:4)e:1,(3:2, 9:1)f:1), ((8:1, 5:1, 6:1)g:1, 7:1)h:1);', format=1)
t3 = Tree('(((1:4,2:4)i:1,(3:2, 9:1)j:1), ((4:3, 5:1, 6:1)k:1, 3:1)l:1);', format=1)
s=t1.get_ascii(show_internal=True)
print(s)
tp = TreePair(t1,t3)

# tp.find_sim_nodes()

# testing component similarity
comp_similarity_matrix = [[1.0,0.8,0.7,0.3,0.2,0.5,0.0,0.1, 0],
                          [0.8,1.0,0.0,0.1,0.3,0.4,0.5,0.0, 0],
                          [0.7,0.0,1.0,0.0,0.5,0.6,0.8,0.0, 0],
                          [0.3,0.1,0.0,1.0,0.3,0.4,0.2,0.9, 0],
                          [0.2,0.3,0.5,0.3,1.0,0.0,0.5,0.6, 0],
                          [0.5,0.4,0.6,0.4,0.0,1.0,0.5,0.6, 0],
                          [0.0,0.5,0.8,0.2,0.5,0.5,1.0,0.3, 0],
                          [0.1,0.0,0.0,0.9,0.6,0.6,0.3,1.0, 0],
                          [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 1]]
tp.find_sim_nodes(True, comp_similarity_matrix)

tc = TreeCompare(t1, [t2, t3])
tc.find_same_nodes()

#'''Visualization'''
# Basic tree style
# ts = TreeStyle()
# ts.show_leaf_name = True
# ts.show_branch_length = True

# Creates an independent node style for each node, which is
# initialized with a red foreground color.
# for n in t1.traverse():
#    nstyle = NodeStyle()
#    nstyle["fgcolor"] = "red"
#    nstyle["size"] = 15
#    n.set_style(nstyle)
# for n in t2.traverse():
#    nstyle = NodeStyle()
#    nstyle["fgcolor"] = "red"
#    nstyle["size"] = 15
#    n.set_style(nstyle)


# cm1,cm2, cm12=tp.find_same_nodes()
# #plot the common nodes green
# for l in [cm1, cm2]:
#     for node in l:
#         node.img_style["size"] = 20
#         node.img_style["fgcolor"] = "green"
#         for child in node.search_nodes():
#             child.img_style["size"] = 20
#             child.img_style["fgcolor"] = "green"


# t1.show(tree_style=ts) # requires PyQt5
# t2.show(tree_style=ts) # requires PyQt5