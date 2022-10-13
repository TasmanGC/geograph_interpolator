from geograph_interpolator import *
import sys
import os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# sine_wave basment depth
def z_function(x, y, d_var=10, f_var=0.1,elevation=150):
    return (np.sin(np.sqrt(x ** 2 + y ** 2)*f_var)*d_var)+elevation

if __name__ == '__main__':
    # define some dummy lines, stations and investigation depths
    lines   = np.linspace(-1000, 1000, 10)         # we have 10 lines
    statn   = np.linspace(-1000, 1000, 100)        # each line has 100 stations
    depths  = np.linspace(-100,0,30)               # each station has 50 depths

    # set of core nodes in 3D space with a label
    core_node_list = []
    n_id = 0
    for i,line in enumerate(lines):
        for stat in statn:
            for d in depths:
                nloc = (line, stat, d)     

                params = {}
                params['line'] = f'L_{str(i).zfill(2)}'
                params['stat'] = f'L_{str(int(i)).zfill(2)}_{str(int(stat)).zfill(4)}'
                params['base'] = z_function(line, stat, d_var=20,f_var=0.01,elevation=-50)
                params['role'] = 0 if i%2==0 else 1

                c_node = CoreNode(n_id,nloc,params=params)
        
                c_node.params['labl'] = 1 if c_node.nloc[2] < c_node.params['base'] else 0        # basment and cover
                # create some dummy conductivity values
                params['cond'] = (c_node.params['labl']*(-20*c_node.nloc[2])) + 100 + np.random.normal(300,200,1)[0]

                core_node_list.append(c_node)
                n_id =+ 1
    
    # generate connections
    ## minkowski distance method very slow
    # mink_filename = 'mink_distance_matrix.csv'
    # mink_calc(node_set,mink_filenam)
    # sparse_similarity = mink_sparse(mink_filenam,core_node_list,n_cons=2,save_name='graph_edges_similarity.csv')
    # e_weights_similarity = edge_weight(core_node_list, sparse_similarity)
    # sparse_similarity['W'] = [x[0] for x in e_weights_similarity]
    # sparse_similarity.to_csv('graph_edges_similarity_weights.csv')

    ## lattice method fast assumes 30 Depth stations and that data has a line parameter
    # sparse_lattice = lattice_sparse(core_node_list,max_cross=1,save_name='graph_edges_lattice.csv')
    # e_weights_lattice = edge_weight(core_node_list, sparse_lattice)
    # sparse_lattice['W'] = [x[0] for x in e_weights_lattice]
    # sparse_lattice.to_csv('graph_edges_lattice_weights.csv')

    # create graph_feat_csv
    # node_feats = {}
    # node_feats['X']       = [x.nloc[0] for x in core_node_list]
    # node_feats['Y']       = [x.nloc[1] for x in core_node_list]
    # node_feats['Z']       = [x.nloc[2] for x in core_node_list]
    # node_feats['C']       = [x.params['cond'] for x in core_node_list]
    # node_feats['LINE']    = [x.params['line'] for x in core_node_list]
    # node_feats['L']       = [x.params['labl'] for x in core_node_list]
    # node_feats['R']       = [x.params['role'] for x in core_node_list]
    # node_feat_df          =  pd.DataFrame.from_dict(node_feats)
    # node_feat_df.to_csv('graph_features.csv')