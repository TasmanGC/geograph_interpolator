# import random
import numpy as np
# import pandas as pd
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import pandas as pd
from .data_node import CoreNode
#from geograph_interpolator.data_processing.data_structures import CoreNode
# import os
# import imageio
# import pyvista as pv
# from pathlib import Path

# def load_xyz(file_name):
#     """ Opens a plain text file with x,y,z values and returns 3 lists.
#     Args:
#         file_name (str) : an absolute directory to a .xyz file.

#     Returns:
#         x_list (list)   : a list of x values.
#         y_list (list)   : a list of y values.
#         z_list (list)   : a list of z values.

#     """ 
#     with open(file_name) as f:
#         x_list = []
#         y_list = []
#         z_list = []

#         for line in f:
#             x,y,z = line.split()
#             x_list.append(float(x))
#             y_list.append(float(y))
#             z_list.append(float(z))
#         return(x_list,y_list,z_list)

# def xyz_surf(x,y,z):
#     """ Converts list of x,y,z values into a mesh file in pyvista.
#     Args:
#         x (list) : a list of x values.
#         y (list) : a list of y values.
#         z (list) : a list of z values.

#     Returns:
#         surf (pyvista, Polydata)    : a surface for 3D visualisation.

#     """

#     points = list(zip(x,y,z))
#     cloud = pv.PolyData(points)
#     cloud['ELEVATION'] = z
#     surf = cloud.delaunay_2d()
#     return(surf)


# def quantile_trans(array):
#     quantile        =   QuantileTransformer(output_distribution='normal')
#     quant_array     =   quantile.fit_transform(array.reshape(-1, 1)) # transform it into a normal distribution
#     return(quant_array)

def load_graph_nodes(file_name):
    df = pd.read_csv(file_name,index_col='Unnamed: 0')
    core_node_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        loc = [row['X'],row['Y'],row['Z']]
        params = {'labl':row['L'],'role':row['R'],'Line':row['LINE'],'cond':row['C']}
        core_node_list.append(CoreNode(i,loc,params=params))
    return(core_node_list)

def load_graph_edges(file_name):
    df = pd.read_csv(file_name)
    
    # weights = df['W'].values
    # df.drop('W',axis=1,inplace=True)
    return(df)

def min_max_norm(array):
    array = np.array(array).reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(array)
    return(scaler.transform(array))

# def gen_random_cons(num_nodes,num_cons):
#     u = []
#     v = []

#     for i in range(num_nodes):
#         u_i = [i]*num_cons
#         v_i = random.sample(list(range(num_nodes)),num_cons)
#         u.extend(u_i)
#         v.extend(v_i)

#     uv_dict = {}
#     uv_dict["U"] = u
#     uv_dict["V"] = v

#     con_df = pd.DataFrame.from_dict(uv_dict)
#     return(con_df)

# def load_xyz(file_name):   
#     with open(file_name) as f:
#         x_list = []
#         y_list = []
#         z_list = []

#         for line in f:
#             x,y,z = line.split()
#             x_list.append(float(x))
#             y_list.append(float(y))
#             z_list.append(float(z))
#         return(x_list,y_list,z_list)

# def dir2gif(dir,save_file):
#     full_paths = []
#     for dirpath,_,filenames in os.walk(dir):
#         for f in filenames:
#             full_paths.append(Path(os.path.abspath(os.path.join(dirpath, f))))
#     images = []
#     for filename in full_paths:
#         images.append(imageio.imread(filename))
#     imageio.mimsave(Path(dir).joinpath(Path(save_file)), images)