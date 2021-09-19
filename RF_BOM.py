import PyQt5
from ete3 import Tree #, NodeStyle, TreeStyle
import re
from pulp import *
import numpy as np

'''This programm is the implementation and expansion of RF-BOM algorithm'''


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
    def __init__(self, t1, t2, sachn_formenschl=None, consider_comp_similarity=False):
        self.t1=t1
        self.t2=t2
        self.sachn_formenschl = sachn_formenschl

        # get all the nodes of tree 1 and tree 2
        t1_all_nodes = get_nodes_with_hierachy_level(t1,[(t1,0)],0)
        t2_all_nodes = get_nodes_with_hierachy_level(t2,[(t2,0)],0)

        # get all the leafs of tree 1 and tree 2
        t1_comp = [x[0].name for x in t1_all_nodes if x[0].is_leaf()==True]
        t2_comp = [x[0].name for x in t2_all_nodes if x[0].is_leaf()==True]

        # get the union of all the leafs of tree 1 and tree 2
        all_comp = list(set(t1_comp)|set(t2_comp))
        all_comp.sort()

        self.all_comp = all_comp


        

        # all the inner nodes of tree 1 and tree 2
        self.t1_inner_nodes = [x for x in t1_all_nodes if x[0].is_leaf()==False]
        self.t2_inner_nodes = [x for x in t2_all_nodes if x[0].is_leaf()==False]

        t1_dic={}
        t2_dic={}
        '''fill the dictionaries:
        the dictionary will be have TreeNode objects of inner nodes as keys and sub-dictionaries as value
        sub-dictionaries will habe leaf names as keys and a tuple (relative hierarchy level, quantity of the leaf) as value
        relative hierarchy level = hierarchy level of the leaf - hierarchy level of the inner node'''
        '''TreeNode.dist describes the quantity of the node instead of distance to the branch!'''
        for node in self.t1_inner_nodes:
            node_dic={}
            for x in get_nodes_with_hierachy_level(node[0],[],node[1]):
                if x[0].is_leaf():
                    rel_hier_level = x[1]-node[1]
                    quantity = x[0].dist
                    quantity *= x[0].up.dist if x[0].up != node[0] else 1 # consider the case that the parent node appears x>1 times
                    node_dic[x[0].name]=(rel_hier_level,quantity) # (relative hierarchy level, quantity)

            t1_dic[node[0]]=node_dic

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
            mat1[n+1]=[1/t1_dic.get(self.t1_inner_nodes[n][0]).get(x)[0]*t1_dic.get(self.t1_inner_nodes[n][0]).get(x)[1] if x in [x.name for x in self.t1_inner_nodes[n][0].search_nodes()] else 0 for x in all_comp]
        mat2={}
        for n in range(0, len(self.t2_inner_nodes)):
            mat2[n+1]=[1/t2_dic.get(self.t2_inner_nodes[n][0]).get(x)[0]*t2_dic.get(self.t2_inner_nodes[n][0]).get(x)[1] if x in [x.name for x in self.t2_inner_nodes[n][0].search_nodes()] else 0 for x in all_comp]

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
        same_nodes_t1=[] # stores the identical nodes that t1 has
        same_nodes_t2=[] # stores the identical nodes that t2 has
        same_nodes_t1_t2=[] # stores the identical nodes in t1 and t2 as tuples
        # find the identical rows in the matrixes of t1 and t2
        common_node_values=[x for x in list(mat1.values()) if x in list(mat2.values())]
        for common_node_value in common_node_values:
            # find the number of the identical node in t1 (not the index)
            common_node_t1=list(mat1.keys())[list(mat1.values()).index(common_node_value)]
            # find the number of the identical node in t2 (not the index)
            common_node_t2=list(mat2.keys())[list(mat2.values()).index(common_node_value)]
            # insert the nodes into the lists by using their index (index = number - 1)
            same_nodes_t1.append(nodes1[common_node_t1-1][0])
            same_nodes_t2.append(nodes2[common_node_t2-1][0])
            same_nodes_t1_t2.append((nodes1[common_node_t1-1][0], nodes2[common_node_t2-1][0]))
        return same_nodes_t1, same_nodes_t2, same_nodes_t1_t2

    '''returns the cosine similarity of two Formenschlüssel (Strings)'''
    def cosine_similarity(self, fm1, fm2):
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

    '''returns the similarity matrix based on all the formenschlüssel'''
    def construct_sim_matrix(self):
        sachn_formenschl=self.sachn_formenschl
        if sachn_formenschl is None:
            return None
        sim_matrix=[]
        for bt in self.all_comp:
            lst=[]
            fs = sachn_formenschl.get(bt)
            for bt2 in self.all_comp:
                if bt2 == bt:
                    lst.append(1)
                else:
                    fs2 = sachn_formenschl.get(bt2)
                    if fs2 is not None and fs is not None:
                        lst.append(self.cosine_similarity(fs,fs2))
                    else:
                        lst.append(0)
            sim_matrix.append(lst)
        return sim_matrix

    ''' returns (d1, d2), where d1 is the distance between node 1 and node 2 while mat1 is mapped to 
    mat2 and d2 is the distance between node 1 and node 2 while mat2 is mapped to mat1'''
    def cal_distance (self, node1, node2, map_1_to_2, consider_comp_similarity=False, comp_similarity_matrix=None ):
        mat1=self.mat1
        mat2=self.mat2
        t1_inner_nodes = self.t1_inner_nodes
        t2_inner_nodes = self.t2_inner_nodes
        t1_dic=self.t1_dic
        t2_dic=self.t2_dic
        all_comp = self.all_comp
        n_components=len(self.all_comp)


        if map_1_to_2: 
            l1 = [mat1[node1][x]-mat2[node2][x] for x in range (0, n_components)]
            dist_final = sum([d for d in l1 if d>=0])
        else:
            l2 = [mat2[node2][x]-mat1[node1][x] for x in range (0, n_components)]
            dist_final =  sum([d for d in l2 if d>=0])
        '''if component similarities are to be considered: use another interger programming model to find out
        the optimal component to component matching and substract the total similarities of matched components from 
        the distance between node 1 and node 2 while mat1 is mapped to mat2 (and the other way around)'''
        if consider_comp_similarity and comp_similarity_matrix is not None:
            #	- t12: (4, 4, 1, 0, 0, 0, 0, 0, 0)
	        #   - t24: (0, 0, 2, 0, 0, 0, 0, 0, 1)
            t1n=[t1_dic.get(t1_inner_nodes[node1-1][0]).get(x)[1] if x in [y.name for y in t1_inner_nodes[node1-1][0].search_nodes()] else 0 for x in all_comp]
            t2n=[t2_dic.get(t2_inner_nodes[node2-1][0]).get(x)[1] if x in [y.name for y in t2_inner_nodes[node2-1][0].search_nodes()] else 0 for x in all_comp]
            ind_t1n_diff_t2n = [x for x in range(0, len(t1n)) if t2n[x]<t1n[x]] # indexes of all components in t1n that are not in t2n
            t1n_minus_t2n_positive = [t1n[i]-t2n[i] if t1n[i]>t2n[i] else 0 for i in range(0, len(t1n))] # list of (t1n - t2n)>=0
            ind_t2n_diff_t1n = [x for x in range(0, len(t2n)) if t2n[x]>t1n[x]]
            t2n_minus_t1n_positive = [t2n[i]-t1n[i] if t1n[i]<t2n[i] else 0 for i in range(0, len(t1n))]
            
            # ind_t1n_diff_t2n: [0, 1]
            # ind_t2n_diff_t1n: [2, 8]
            # t12-t24: (4, 4, 0, 0, 0, 0, 0, 0, 0)
            # t24-t12: (0, 0, 1, 0, 0, 0, 0, 0, 1)
            if map_1_to_2:
                prob = LpProblem("component_matching_problem1", LpMaximize)
                pvar1 = [[LpVariable("t1.replace{}by{}".format(i+1, j+1), 0, None, LpInteger) for j in ind_t2n_diff_t1n ] for i in ind_t1n_diff_t2n]            

                prob+= sum([sum([pvar1[ind_t1n_diff_t2n.index(i)][ind_t2n_diff_t1n.index(j)]*comp_similarity_matrix[i][j] for j in ind_t2n_diff_t1n])\
                /t1_dic.get(self.t1_inner_nodes[node1-1][0]).get(all_comp[i])[0] for i in ind_t1n_diff_t2n])
                
                constraints_t1 = [sum([pvar1[ind_t1n_diff_t2n.index(i)][ind_t2n_diff_t1n.index(j)] for j in ind_t2n_diff_t1n]) <= t1n_minus_t2n_positive[i] for i in ind_t1n_diff_t2n]
                constraints_t2 = [sum([pvar1[ind_t1n_diff_t2n.index(i)][ind_t2n_diff_t1n.index(j)] for i in ind_t1n_diff_t2n]) <= t2n_minus_t1n_positive[j] for j in ind_t2n_diff_t1n]
    

                for cst in constraints_t1:
                    prob += cst
                for cst in constraints_t2:
                    prob += cst
                

                prob.solve(solver = PULP_CBC_CMD(msg=0))
                
                dist_final -= value(prob.objective) if value(prob.objective) is not None else 0
                if dist_final<0:                
                    print("wrong calculation when considering BT similarity")
                    dist_final=0

            else:
                '''repeat the same procedure for the other way around '''

                prob2 = LpProblem("component_matching_problem2", LpMaximize)
                pvar2 = [[LpVariable("t2.replace{}by{}".format(i+1, j+1), 0, None, LpInteger) for j in ind_t1n_diff_t2n ] for i in ind_t2n_diff_t1n]            

                prob2 += sum([sum([pvar2[ind_t2n_diff_t1n.index(i)][ind_t1n_diff_t2n.index(j)]*comp_similarity_matrix[i][j] for j in ind_t1n_diff_t2n])\
                /t2_dic.get(self.t2_inner_nodes[node2-1][0]).get(all_comp[i])[0] for i in ind_t2n_diff_t1n])
                
                constraints_t1 = [sum([pvar2[ind_t2n_diff_t1n.index(i)][ind_t1n_diff_t2n.index(j)] for j in ind_t1n_diff_t2n]) <= t2n_minus_t1n_positive[i] for i in ind_t2n_diff_t1n]
                constraints_t2 = [sum([pvar2[ind_t2n_diff_t1n.index(i)][ind_t1n_diff_t2n.index(j)] for i in ind_t2n_diff_t1n]) <= t1n_minus_t2n_positive[j] for j in ind_t1n_diff_t2n]
    

                for cst in constraints_t1:
                    prob2 += cst
                for cst in constraints_t2:
                    prob2 += cst
                

                prob2.solve(solver = PULP_CBC_CMD(msg=0))
                
                dist_final -= value(prob2.objective) if value(prob2.objective) is not None else 0
                if dist_final<0:                
                    print("wrong calculation when considering BT similarity")
                    dist_final=0

        return round(dist_final,2)

    '''creates distance mapping matrixes as nested lists using cal_distance'''
    def create_dist_mapping_matrix (self, consider_comp_similarity=False ):
        mat1=self.mat1
        mat2=self.mat2
        n_nodes_mat1 = len(mat1)
        n_nodes_mat2 = len(mat2)
        if not consider_comp_similarity:
            T12 = [[self.cal_distance(i, j, map_1_to_2=True) for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
            T21 = [[self.cal_distance(j, i, map_1_to_2=False) for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        else:
            comp_similarity_matrix = self.construct_sim_matrix()
            T12 = [[self.cal_distance(i, j, True, True, comp_similarity_matrix) for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
            T21 = [[self.cal_distance(j, i, True, True, comp_similarity_matrix) for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        

        return T12, T21

    '''computes the normalizes rf similarity between two tree matrixes using an interger programming model:
        input = the two matrixes obtained from generate_matrixes(t1, t2)
        output: normalized rf metric, values of constraint variables'''  
    def compute_rf_bom (self, consider_comp_similarity=False ):
        mat1=self.mat1
        mat2=self.mat2
        n_nodes_mat1 = len(mat1)
        n_nodes_mat2 = len(mat2)

        # Create the 'prob' variable to contain the problem data
        prob = LpProblem("Node_Matching_Problem", LpMinimize)

        # The problems variables
        t1 = [[LpVariable("z{}{}".format(i, j), 0, 1, LpInteger) for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
        t2 = [[LpVariable("w{}{}".format(i, j), 0, 1, LpInteger) for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        


        if not consider_comp_similarity:
            dm1, dm2 = self.create_dist_mapping_matrix()
        else:
            dm1, dm2 = self.create_dist_mapping_matrix(True)

        RF_BOM_max =sum([sum(mat1[i]) for i in range(1, n_nodes_mat1+1)])+sum([sum(mat2[i]) for i in range(1, n_nodes_mat2+1)])

        node_matching_matrix1 = []
        node_matching_matrix2 = []

        
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

        prob.solve(solver = PULP_CBC_CMD(msg=0))

     

        for v in prob.variables():
            if v.name[0]=='z' and v.varValue==1:
                node_matching_matrix1.append((int(v.name[1]), int(v.name[2])))
            if v.name[0]=='w' and v.varValue==1: 
                node_matching_matrix2.append((int(v.name[1]), int(v.name[2])))

        RF_BOM = value(prob.objective)


        if sum( sum(x) if isinstance(x, list) else x for x in dm1 ) + sum( sum(x) if isinstance(x, list) else x for x in dm2 ) == 0:
            RF_BOM = 0

        if RF_BOM is None:
            RF_BOM_norm = -1
        else:
            RF_BOM_norm = (RF_BOM_max-RF_BOM)/RF_BOM_max
            
        return RF_BOM_norm, node_matching_matrix1, node_matching_matrix2

    '''returns a dictionary with tuples (i, j) as keys and rf-distance between node i of tree 1 and node j of tree 2 as values'''
    def find_sim_nodes(self, consider_comp_similarity=False ):
        t1=self.t1
        t2=self.t2
        mat1=self.mat1
        mat2=self.mat2
        nodes1=self.t1_inner_nodes
        nodes2=self.t2_inner_nodes
        tree_dist, nmm1, nmm2 = self.compute_rf_bom() if not consider_comp_similarity else self.compute_rf_bom(True)
        mapped_nodes = list(set(nmm1).union(set(nmm2)))
        nodes_rf_boms = {}
        for pair in mapped_nodes:
            try:
                n1 = nodes1[pair[0]-1][0]
                n2 = nodes2[pair[1]-1][0]
            except:
                n1 = nodes1[pair[1]-1][0]
                n2 = nodes2[pair[0]-1][0]

            tp = TreePair(n1,n2, self.sachn_formenschl)
            mat_n1=tp.mat1
            mat_n2=tp.mat2
            rf_dist = tp.compute_rf_bom()[0] if not consider_comp_similarity else tp.compute_rf_bom(True)[0]
            n1_name = n1.name
            n2_name = n2.name

            if rf_dist>=0 and rf_dist<=1 and n1_name!= n2_name:
                nodes_rf_boms[(n1_name, n2_name)]=rf_dist
        

        return dict(sorted(nodes_rf_boms.items(), key=lambda item: item[1] , reverse=True))

'''TreeCompare object consists of a tree (self.t) that is to be compared with a list of other different trees (self.trees)'''
class TreeCompare:
    def __init__(self, t, trees, sachn_formenschl=None):
        self.trees = trees
        self.t = t
        self.sachn_formenschl=sachn_formenschl

    def find_same_nodes(self):
        same_nodes = []
        for tree in self.trees:
            tp = TreePair(self.t, tree, self.sachn_formenschl)
            same_nodes.append(tp.find_same_nodes()[2])
        return same_nodes
        
    def find_similar_nodes(self,consider_comp_similarity=False ):
        similar_nodes = []
        for tree in self.trees:
            tp = TreePair(self.t, tree, self.sachn_formenschl)
            sim_nodes = tp.find_sim_nodes(consider_comp_similarity  )
            similar_nodes.append(sim_nodes)
        return similar_nodes
    
    def find_distances(self,consider_comp_similarity=False ):
        distances = []
        for tree in self.trees:
            tp = TreePair(self.t, tree, self.sachn_formenschl)
            tree_dist, nmm1, nmm2 = tp.compute_rf_bom(consider_comp_similarity  )
            distances.append(tree_dist)
        return distances



# sachn_formenschl = {'BT 011': '23355', 'S 003': '14111', 'S 001': '14111', 'BT 014': '21131', 'S 002': '15131', 'BT 017': '12140', 'BT 015': '12140', 'BT 018': '11132', 'BT 001': '12140', 'BT 002': '12140', 'BT 013': '23255', 'BT 019': '11322', 'BT 005': '11111', 'BT 016': '12140', 'BT 006': '11111', 'BT 007': '11111', 'BT 008': '11111'}
# t1 = Tree('((S 001:1,((BT 010:1,BT 011:1,S 002:8)BG 006:1,(BT 005:1,S 001:2)BG 001:1,BT 001:1)BG 011:1,((BT 009:1,S 003:2)BG 010:1,(BT 012:1,S 001:4)BG 007:1,BT 003:1)BG 008:1):1);', format=1)
# t2 = Tree('((S 001:1,((BT 010:1,BT 011:1,S 002:8)BG 006:1,(BT 005:1,S 001:2)BG 001:1,BT 001:1)BG 011:1,((BT 009:1,S 003:2)BG 010:1,(BT 012:1,S 001:4)BG 007:1,(BT 004:1,S 001:3)BG 005:1)BG 009:1):1);', format=1)
# t3 = Tree('((S 001:1,((BT 010:1,BT 011:1,S 002:8)BG 006:1,(BT 006:1,S 001:2)BG 002:1,BT 001:1)BG 012:1,((BT 009:1,S 003:2)BG 010:1,(BT 012:1,S 001:4)BG 007:1,BT 003:1)BG 008:1):1);', format=1)
