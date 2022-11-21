import  numpy as np
import  dgl
import  torch
from    PIL import Image
from    .data_node import CoreNode
from    .support_func import min_max_norm
import  pandas as pd
     

def image2graph(image_fn, include_diag = True, up_factor = 1,return_df=False): # for the time being up_factor will need to be a positive float

    im                  = Image.open(image_fn)  # read a tif
    imarray             = np.array(im)          # node features
    im_width, im_height = imarray.shape         # image dimensions
    up_width, up_height = (v*up_factor for v in imarray.shape)

    x_loc = []
    y_loc = []
    z_loc = []
    n_val = []
    t_val = []

    inc_x = list(range(up_width))[0::up_factor]
    inc_y = list(range(up_height))[0::up_factor]

    # features
    for row_id in list(range(up_width)):
        for col_id in list(range(up_height)):
            x_loc.append(row_id)
            y_loc.append(col_id)
            z_loc.append(1)
            if row_id in inc_x and col_id in inc_y:
                n_val.append(imarray[int(row_id/up_factor)][int(col_id/up_factor)])
                t_val.append(1)
            else:
                n_val.append(0)
                t_val.append(0)

    # edges
    U = []
    V = []

    for node_id in range(up_width*up_height):
        node_cons = []

        # lattice connections
        # adjacent pixels in neighboring rows
        node_cons.extend([node_id+1,node_id-1])
        # adjacent pixels in neighboring columns
        node_cons.extend([node_id+up_width, node_id-up_width])

        if include_diag:
            # diagonal connections
            # adjacent pixels in neighboring rows
            node_cons.extend([node_id+up_width+1,node_id+up_width-1])
            node_cons.extend([node_id-up_width+1,node_id-up_width-1])

        node_cons = [x for x in node_cons if x>0]
        node_cons = [x for x in node_cons if x < (up_width*up_height)]
        node_id_list = len(node_cons)*[node_id]

        U.extend(node_id_list)
        V.extend(node_cons)

    n_val = min_max_norm(n_val)

    graph = dgl.graph((U,V))
    graph.ndata['n_dat']    = torch.tensor(n_val,dtype=torch.float32)
    graph.ndata['x_loc']    = torch.tensor(x_loc)
    graph.ndata['y_loc']    = torch.tensor(y_loc)
    graph.ndata['z_loc']    = torch.tensor(z_loc)
    graph.ndata['t_val']    = torch.tensor(t_val)
    graph = dgl.add_self_loop(graph)
    if return_df:
        df_dict = {}
        df_dict['U'] = U
        df_dict['V'] = V
        df = pd.DataFrame.from_dict(df_dict)
        return(graph,df)
    else:
        return(graph)  

def construct_standard_nodes(graph):
    core_node_list = []
    counter = 0

    n_dat = graph.ndata['n_dat'].numpy()
    x_loc = graph.ndata['x_loc'].numpy()
    y_loc = graph.ndata['y_loc'].numpy()
    z_loc = graph.ndata['z_loc'].numpy()
    t_val = graph.ndata['t_val'].numpy()
    if 'predv' in list(graph.ndata.keys()):
        pred_v = graph.ndata['predv'].numpy()
    
    for i,d in enumerate(n_dat):
        pdict = {'data':d,'role':t_val[i]}
        if 'predv' in list(graph.ndata.keys()):
            pdict['pred'] = pred_v[i]
        core_node  = CoreNode(counter,[x_loc[i],y_loc[i],z_loc[i]],params=pdict)
        core_node_list.append(core_node)
        counter = counter+1
    return(core_node_list)