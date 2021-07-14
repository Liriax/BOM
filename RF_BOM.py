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
    def __init__(self, t1, t2, sachn_formenschl=None, consider_comp_similarity=False):
        self.t1=t1
        self.t2=t2
        self.sachn_formenschl = sachn_formenschl

        t1_all_nodes = get_nodes_with_hierachy_level(t1,[(t1,0)],0)
        t2_all_nodes = get_nodes_with_hierachy_level(t2,[(t2,0)],0)

        

        
        t1_comp = [x[0].name for x in t1_all_nodes if x[0].is_leaf()==True]
        t2_comp = [x[0].name for x in t2_all_nodes if x[0].is_leaf()==True]

        # if consider_comp_similarity:
        #     same_comps = []
        #     for search_fs in set(self.sachn_formenschl.values()):
        #         comps = [bt for bt in sachn_formenschl.keys() if sachn_formenschl.get(bt) == search_fs]
        #         same_comps.append(comps)
        #     same_comp = [x for x in same_comps if len(x)>1]
            
        #     # same_comp = [x for x in [(x,y) if sim_matrix[x][y]==1 else None for x in range(0, leng) for y in range(0, leng)] if x is not None and x[0]!=x[1]]            
        #     for lst in same_comp:
        #         t1_comp = [lst[0] if x in lst else x for x in t1_comp]
        #         t2_comp = [lst[0] if x in lst else x for x in t2_comp]

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

    def construct_sim_matrix(self):
        sachn_formenschl=self.sachn_formenschl
        if sachn_formenschl is None:
            return None
        sim_matrix=[]
        bt_with_fs = sachn_formenschl.keys()
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
    def cal_distance (self, node1, node2, consider_comp_similarity=False, comp_similarity_matrix=None ):
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
            
            # print(value(prob.objective))
            dist1 -= value(prob.objective) if value(prob.objective) is not None else 0
            if dist1<0:                
                print("wrong calculation when considering BT similarity")
                dist1=0


            '''-------------'''
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
            
            # print(value(prob.objective))
            dist2 -= value(prob2.objective) if value(prob2.objective) is not None else 0
            if dist2<0:                
                print("wrong calculation when considering BT similarity")
                dist2=0

            # for vl in pvar1:
            #     for v in vl:
            #         if v.varValue>0:
            #             replaced = int(v.name.split(",")[0])-1
            #             replacer = int(v.name.split(",")[1])-1
            #             mat1[node1][replaced]-=v.varValue*comp_similarity_matrix[replaced][replacer]/t1_dic.get(self.t1_inner_nodes[node1-1][0]).get(all_comp[replaced])[0]
            #             mat1[node1][replacer]+=v.varValue*comp_similarity_matrix[replaced][replacer]/t1_dic.get(self.t1_inner_nodes[node1-1][0]).get(all_comp[replaced])[0]

        return round(dist1,2), round(dist2,2)

    '''creates distance mapping matrixes as nested lists using cal_distance'''
    def create_dist_mapping_matrix (self, consider_comp_similarity=False ):
        mat1=self.mat1
        mat2=self.mat2
        n_nodes_mat1 = len(mat1)
        n_nodes_mat2 = len(mat2)
        if not consider_comp_similarity:
            T12 = [[self.cal_distance(i, j)[0] for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
            T21 = [[self.cal_distance(j, i)[1] for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        else:
            comp_similarity_matrix = self.construct_sim_matrix()
            T12 = [[self.cal_distance(i, j, True, comp_similarity_matrix )[0] for j in range(1, n_nodes_mat2+1)] for i in range(1, n_nodes_mat1+1)]
            T21 = [[self.cal_distance(j, i, True, comp_similarity_matrix )[1] for j in range(1, n_nodes_mat1+1)] for i in range(1, n_nodes_mat2+1)]
        

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

        # prob.writeLP("NodeMatching.lp")

        # The problem is solved using PuLP's choice of Solver
        prob.solve(solver = PULP_CBC_CMD(msg=0))

        # The status of the solution is printed to the screen
        # print("Status:", LpStatus[prob.status])

        # Each of the variables is printed with it's resolved optimum value
        # for v in prob.variables():
        #     print(v.name, "=", v.varValue)

        for v in prob.variables():
            if v.name[0]=='z' and v.varValue==1:
                node_matching_matrix1.append((int(v.name[1]), int(v.name[2])))
            if v.name[0]=='w' and v.varValue==1: 
                node_matching_matrix2.append((int(v.name[1]), int(v.name[2])))

        RF_BOM = value(prob.objective)
        # The optimised objective function value is printed to the screen
        # print("Total node matching distance = ", value(prob.objective))

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
        # print(mapped_nodes)
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
            rf_dist = tp.compute_rf_bom()[0] if not consider_comp_similarity else tp.compute_rf_bom(True  )[0]
            n1_name = n1.name
            n2_name = n2.name
            # print(n1.name, n2.name, rf_dist)


            if rf_dist>=0 and rf_dist<=1 and n1_name!= n2_name:
                nodes_rf_boms[(n1_name, n2_name)]=rf_dist
        

        return dict(sorted(nodes_rf_boms.items(), key=lambda item: item[1] , reverse=True))


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



sachn_formenschl = {'BT 011': '23355', 'S 003': '14111', 'S 001': '14111', 'BT 014': '21131', 'S 002': '15131', 'BT 017': '12140', 'BT 015': '12140', 'BT 018': '11132', 'BT 001': '12140', 'BT 002': '12140', 'BT 013': '23255', 'BT 019': '11322', 'BT 005': '11111', 'BT 016': '12140', 'BT 006': '11111', 'BT 007': '11111', 'BT 008': '11111'}
t1 = Tree('((S 001:1,((BT 010:1,BT 011:1,S 002:8)BG 006:1,(BT 005:1,S 001:2)BG 001:1,BT 001:1)BG 011:1,((BT 009:1,S 003:2)BG 010:1,(BT 012:1,S 001:4)BG 007:1,BT 003:1)BG 008:1):1);', format=1)
t2 = Tree('((S 001:1,((BT 010:1,BT 011:1,S 002:8)BG 006:1,(BT 005:1,S 001:2)BG 001:1,BT 001:1)BG 011:1,((BT 009:1,S 003:2)BG 010:1,(BT 012:1,S 001:4)BG 007:1,(BT 004:1,S 001:3)BG 005:1)BG 009:1):1);', format=1)
t3 = Tree('((S 001:1,((BT 010:1,BT 011:1,S 002:8)BG 006:1,(BT 006:1,S 001:2)BG 002:1,BT 001:1)BG 012:1,((BT 009:1,S 003:2)BG 010:1,(BT 012:1,S 001:4)BG 007:1,BT 003:1)BG 008:1):1);', format=1)
# t1 = Tree('(BT 005:1,S 001:2)BG 001:1;',format=1)
# t3 = Tree('(BT 006:1,S 001:2)BG 002:1;',format=1)

# s=t1.get_ascii(show_internal=True)
# print(s)
# tp = TreePair(t1,t3)
# print(tp.compute_rf_bom())
# print(tp.find_sim_nodes())
# tp = TreePair(t1,t3, sachn_formenschl, True)
# print(tp.compute_rf_bom(True))
# print(tp.find_sim_nodes(True))
# # tp.find_sim_nodes()

# # testing component similarity
# comp_similarity_matrix = [[1.0,0.8,0.7,0.3,0.2,0.5,0.0,0.1, 0],
#                           [0.8,1.0,0.0,0.1,0.3,0.4,0.5,0.0, 0],
#                           [0.7,0.0,1.0,0.0,0.5,0.6,0.8,0.0, 0],
#                           [0.3,0.1,0.0,1.0,0.3,0.4,0.2,0.9, 0],
#                           [0.2,0.3,0.5,0.3,1.0,0.0,0.5,0.6, 0],
#                           [0.5,0.4,0.6,0.4,0.0,1.0,0.5,0.6, 0],
#                           [0.0,0.5,0.8,0.2,0.5,0.5,1.0,0.3, 0],
#                           [0.1,0.0,0.0,0.9,0.6,0.6,0.3,1.0, 0],
#                           [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 1]]
# tp.find_sim_nodes(True  )

# tc = TreeCompare(t1, [t2, t3], sachn_formenschl)
# print(tc.find_similar_nodes(True))
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