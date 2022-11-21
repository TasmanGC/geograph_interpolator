
import csv
import numpy as np
import pandas as pd
import torch
import dgl
import sklearn.metrics as skm
import torch.nn as nn
from tqdm import tqdm
from scipy.spatial.distance import minkowski
import itertools
from scipy.spatial import distance_matrix
import torch.nn.functional as F
# from sklearn.metrics    import f1_score, precision_score, recall_score
# from scipy.interpolate  import LinearNDInterpolator
# import dgl
# import random
# from torch import tensor
# from torch import float32

from .support_func import min_max_norm

def edge_weight(node_list,e_set):
    '''
    e_set dataframe should have only u and v columns
    '''
    m_dist_all = []

    for u,v in e_set.values:
        v_1     = [val for _list in [node_list[u].nloc, [node_list[u].params['cond']]] for val in _list]
        v_2     = [val for _list in [node_list[v].nloc, [node_list[v].params['cond']]] for val in _list]
        #calculate minkowski distance
        m_dist  = minkowski(v_1,v_2,1)
        m_dist_all.append(m_dist)
    
    # lets convert distance to an edge weights
    norm        = min_max_norm(m_dist_all)
    e_weights   = [1-n for n in norm]

    return(e_weights)

def mink_calc(node_set,file_name):
    '''
    Function to calculate a distance matrix used to calculate minkowski distance between all nodes of our Graph
    '''
    for i,n in enumerate(node_set):
        n.params['iD'] = i

    vectors = np.array([np.array([n.nloc[0],n.nloc[1],n.nloc[2]]) for n in node_set])
    
    with open(file_name, "w",newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i,v in tqdm(enumerate(vectors)):
            a = distance_matrix(vectors[i].reshape(1,-1),vectors, p=1)
            writer.writerow(a[0])

def mink_sparse(minkmatrix_path,node_list,n_cons=2,save_name='graph_edges_similarity.csv'):
    '''
    Uses a calculated distance matrix and set of nodes to calculate a sparse adjacency matrix.
    By default,saves sparse matrix as csv, you can set it to none if you just want the dataframe.
    '''
    # unique set of line names
    lines = list(set([x.params['Line'] for x in node_list]))

    # a surprise tool that'll help us later
    for i in range(len(node_list)):
        node_list[i].params['ID_MAN'] = i

    # define our sparse connections
    u = []
    v = []

    with open(minkmatrix_path, newline='') as f: # because these files are typically huge we need to read it one line at a time
        reader = csv.reader(f)

        for i, row in tqdm(enumerate(reader)):
            # should be same number of rows as there are nodes
            i_node = node_list[i]

            ## list comprehensions are used to select all but the current node
            # the minkowski distance to other nodes
            i_mdist = np.array([x for j,x in enumerate(row) if j!=i])
            # the line of all other nodes
            i_line = np.array([x.params['line'] for j,x in enumerate(node_list) if j!=i])
            # the correct node ID of other nodes
            i_onode = np.array([x.params['ID_MAN'] for j,x in enumerate(node_list) if j!=i])

            # connect here to the other lines
            i_v = []
            for l in lines:
                # lets filter some things by lines 
                cline_dis = i_mdist[np.where(i_line==l)]
                cline_ids = i_onode[np.where(i_line==l)]
                
                idx = np.argpartition(cline_dis,n_cons)
                ids = cline_ids[idx[:n_cons]]
                i_v.extend(ids)
            i_u = [i_node.params['ID_MAN']]*len(i_v)

            # extend our overall u-v
            u.extend(i_u)
            v.extend(i_v)

    graph_edge_df = {}
    graph_edge_df['U'] = u 
    graph_edge_df['V'] = v 

    edges = pd.DataFrame.from_dict(graph_edge_df)
    
    if save_name is not None:
        edges.to_csv(save_name)
        return(edges)
    else:
        return(edges)

def lattice_sparse(node_list,max_cross=1,save_name='graph_edges_lattice.csv'):
    for i,n in enumerate(node_list):
        n.params['iD'] = i

    line_list = np.unique([node.params["line"] for node in node_list])

    # some useful parameters in our dataset
    node_per_line   = {x : len([y for y in node_list if y.params['line']==x]) for x in line_list}
    line_stations   = {x : sorted(list(set([(y.nloc[1]) for y in node_list if y.params['line']==x]))) for x in line_list}
    
    # lets sort our nodes 
    sorted_node_list = []
    for line in line_list:
        line_nodes = [x for x in node_list if x.params['line']==line]
        line_nodes.sort(key=lambda x: (x.nloc[1]))
        sorted_node_list.extend(line_nodes)

    # caclulate connections between adjacent lines
    node_l = list(node_per_line.values())
    result = [0] + list(np.cumsum(node_l))

    # use these to make cool stuff
    x = np.array([x.nloc[0] for x in node_list])
    y = np.array([x.nloc[1] for x in node_list])
    z = np.array([x.nloc[2] for x in node_list])
    spatial = np.array(list(zip(x,y,z)))

    # top and bottom
    top = np.arange(0,len(node_list),30,dtype=int)
    bot = np.arange(29,len(node_list),30,dtype=int)

    u=[]
    v=[]


    for i, node in enumerate(tqdm(sorted_node_list)):

        # top or botoom
        t_ = True if i in top else False
        b_ = True if i in bot else False

        # define the cons
        if t_:
            u.extend([i])
            v.extend([i+1])
        if b_:
            u.extend([i])
            v.extend([i-1])
        if not t_ and not b_:
            u.extend([i]*2)
            v.extend([i+1,i-1])

        # # this station and next station
        frst = True if line_stations[node.params['line']][0]==(node.nloc[1]) else False
        last = True if line_stations[node.params['line']][-1] == (node.nloc[1]) else False
        
        # defining cons
        if frst:
            u.extend([i])
            v.extend([i+30])

        if last:
            u.extend([i])
            v.extend([i-30])

        if not frst and not last:
            u.extend([i]*2)
            v.extend([i+30,i-30])

        #previous line and next line
        dist = skm.pairwise.euclidean_distances(X=spatial, Y=spatial[i].reshape(1,-1))

        frst_line = True if list(line_stations)[0]==node.params['line'] else False
        last_line = True if list(line_stations)[-1]==node.params['line'] else False

        if frst_line: # node is in first line
            n_line = list(line_stations)[list(line_stations).index(node.params['line'])+1]
            n_nodes = [n for n in node_list if n.params['line']==n_line]
            n_spati = np.array([x.nloc for x in n_nodes])
            n_ids   = np.array([x.params['iD'] for x in n_nodes])
            dist = skm.pairwise.euclidean_distances(X=n_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            n_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(n_ids[n_closest])

        if last_line: # node is in last line
            p_line = list(line_stations)[list(line_stations).index(node.params['line'])-1]
            p_nodes = [n for n in node_list if n.params['line']==p_line]
            p_spati = np.array([x.nloc for x in p_nodes])
            p_ids   = np.array([x.params['iD'] for x in p_nodes])
            dist = skm.pairwise.euclidean_distances(X=p_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            p_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(p_ids[p_closest])

        if not frst_line and not last_line: # node has two adjacent lines
            p_line = list(line_stations)[list(line_stations).index(node.params['line'])-1]
            p_nodes = [n for n in node_list if n.params['line']==p_line]
            p_spati = np.array([x.nloc for x in p_nodes])
            p_ids   = np.array([x.params['iD'] for x in p_nodes])
            dist = skm.pairwise.euclidean_distances(X=p_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            p_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(p_ids[p_closest])

            n_line = list(line_stations)[list(line_stations).index(node.params['line'])+1]
            n_nodes = [n for n in node_list if n.params['line']==n_line]
            n_spati = np.array([x.nloc for x in n_nodes])
            n_ids   = np.array([x.params['iD'] for x in n_nodes])
            dist = skm.pairwise.euclidean_distances(X=n_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            n_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(n_ids[n_closest])

    # generate a dataframe
    conns = {}
    conns['U'] = u
    conns['V'] = v
    conns = pd.DataFrame.from_dict(conns)
    # screen connections that shouldn't exist
    edges = conns[conns.U >0]
    edges = edges[edges.V >0]
    edges = edges[edges.U < 29999]
    edges = edges[edges.V < 29999]
    if save_name is not None:
        edges.to_csv(save_name)
        return(edges)
    else:
        return(edges)

def create_dgl_graph(nodes,edges):

    # edges - as sets of to and from
    u = edges['U'].astype('int64').values
    v = edges['V'].astype('int64').values
    w = edges['W'].astype('float32').values

    
    xloc    = np.array([n.nloc[0]         for n in nodes],dtype=np.float32).flatten()
    yloc    = np.array([n.nloc[1]         for n in nodes],dtype=np.float32).flatten()
    zloc    = np.array([n.nloc[2]         for n in nodes],dtype=np.float32).flatten()
    
    xloc_n  = np.array(min_max_norm([n.nloc[0]         for n in nodes]),dtype=np.float32).flatten()
    yloc_n  = np.array(min_max_norm([n.nloc[1]         for n in nodes]),dtype=np.float32).flatten()
    zloc_n  = np.array(min_max_norm([n.nloc[2]         for n in nodes]),dtype=np.float32).flatten()
    data    = np.array(min_max_norm([n.params['cond']  for n in nodes]),dtype=np.float32).flatten()
    role    = np.array([n.params['role']    for n in nodes])
    labl    = np.array([n.params['labl']    for n in nodes])

    # generate and populate our bipartite graph
    graph = dgl.graph((u, v),num_nodes=len(nodes))
    graph.edata['W'] = torch.tensor(w)

    graph.ndata['xloc'] = torch.tensor(xloc).unsqueeze(1)
    graph.ndata['yloc'] = torch.tensor(yloc).unsqueeze(1)
    graph.ndata['zloc'] = torch.tensor(zloc).unsqueeze(1)
    graph.ndata['xloc_n'] = torch.tensor(xloc_n).unsqueeze(1)
    graph.ndata['yloc_n'] = torch.tensor(yloc_n).unsqueeze(1)
    graph.ndata['zloc_n'] = torch.tensor(zloc_n).unsqueeze(1)
    graph.ndata['data'] = torch.tensor(data).unsqueeze(1)
    graph.ndata['role'] = torch.tensor(role).unsqueeze(1)  
    graph.ndata['labl'] = torch.tensor(labl).unsqueeze(1)

    graph = dgl.add_self_loop(graph)

    return(graph)

def gnn_interpolate_grid(model, graph, node_feats=5,epochs=10):

    node_embed  = nn.Embedding(graph.num_nodes(),node_feats)

    inputs  = node_embed.weight
    labels  = graph.ndata['n_dat']
    filt    = torch.where(graph.srcdata['t_val']==1,True,False).flatten()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(),node_embed.parameters()), lr=0.01)
    
    all_logits = []
    all___loss = []
    for epoch in tqdm(range(epochs)):

        logits = model(graph,inputs)

        # we save the logits for visualization later
        all_logits.append(logits.detach())
        loss = F.l1_loss(logits[filt], labels[filt])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all___loss.append(loss.item())    
    
    return(all___loss,all_logits)

def gnn_interpolate(model, graph, node_feats=5,epochs=10):

    # node embedding to use
    if isinstance(node_feats,int): # learnable embedding
        node_embed  = nn.Embedding(graph.num_nodes(),node_feats)
    if isinstance(node_feats,list): # real_features
        node_embed          = nn.Embedding(graph.num_nodes(),len(node_feats))
        real_feats          = torch.stack([graph.ndata.get(key).flatten() for key in node_feats],dim=-1)
        node_embed.weight   = torch.nn.Parameter(torch.tensor(real_feats), requires_grad=True)

    inputs  = node_embed.weight
    labels  = torch.tensor([[0,1] if l==0 else [1,0] for l in graph.ndata['labl']],dtype = torch.float32)
    filt    = torch.where(graph.srcdata['role']==1,True,False).flatten()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(),node_embed.parameters()), lr=0.01)
    
    all_logits = []
    all___loss = []
    for epoch in tqdm(range(epochs)):

        logits = model(graph,inputs)

                # we save the logits for visualization later
        all_logits.append(logits.detach())
        loss = F.binary_cross_entropy(logits[filt], labels[filt])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all___loss.append(loss.item())    
    
    return(all___loss,all_logits)

#     def interp_labels(self):
#         xk = np.array([n.nloc[0] for n in self.nodes if n.params['training']==1])
#         yk = np.array([n.nloc[1] for n in self.nodes if n.params['training']==1])
#         zk = np.array([n.nloc[2] for n in self.nodes if n.params['training']==1])
#         vk = np.array([1 if n.params['label']=='cover' else 0 for n in self.nodes if n.params['training']==1])

#         #rbf4 = Rbf(xk, yk, zk, vk, function="multiquadric",smooth=2)#function='linear')
#         linear  = LinearNDInterpolator(list(zip(xk, yk, zk)),vk)

#         xu = np.array([n.nloc[0] for n in self.nodes if n.params['training']==0])
#         yu = np.array([n.nloc[1] for n in self.nodes if n.params['training']==0])
#         zu = np.array([n.nloc[2] for n in self.nodes if n.params['training']==0])
#         ui = np.array([i for i,n in enumerate(self.nodes) if n.params['training']==0])

#         #s_test = rbf4(xu, yu, zu)
#         s_test = linear(xu,yu,zu)
#         #s_test = [0 if x < 0.5 else 1 for x in rbfv]
#         j = 0
#         for i, n in enumerate(self.nodes):
#             n.params['interp'] = 1 if n.params['label']=='cover' else 0
#             if i in ui:
#                 n.params['interp'] =  s_test[j]
#                 j=j+1