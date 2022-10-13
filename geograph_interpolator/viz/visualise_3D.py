
import numpy    as np
import pyvista  as pv
import pandas   as pd
import matplotlib.pyplot as plt


def construct_standard_edges(graph):
    # all relevant connections etc
    src_type = [e[0] for e in graph.canonical_etypes]
    edg_type = [e[1] for e in graph.canonical_etypes]
    dst_type = [e[2] for e in graph.canonical_etypes]

    # we need to making a mapping dictionary between two values
    mapp_dic = {}
    currentid = 0
    for key in ['aem','sei']:
        n_nodes = graph.num_nodes(key)

        type_id = list(range(n_nodes))
        glob_id = list(range(currentid,currentid+n_nodes))

        mapp_dic[key] = {k:v for k,v in zip(type_id,glob_id)}
        
        currentid = currentid + n_nodes

    # now we use the mapping and the various adjaceny matricies to add the edges
    global_src = []
    global_dst = []
    global_clr = []

    color_dict = {k:v for v,k in enumerate(edg_type)}

    for i, e_type in enumerate(edg_type):
        src_type_i = src_type[i]
        dst_type_i = dst_type[i]
        etype_spadj = graph.adj(etype = e_type).coalesce().indices().numpy()

        # use the node type and edge mapping to create the vertices
        for u,v in zip(etype_spadj[0],etype_spadj[1]):
            global_u = mapp_dic[src_type_i][u]
            global_v = mapp_dic[dst_type_i][v]
            global_src.append(global_u)
            global_dst.append(global_v)
            global_clr.append(color_dict[e_type])
    
    con_dict = {'U':global_src,'V':global_dst,'C':global_clr}

    return(pd.DataFrame.from_dict(con_dict))

def visualise_3D(data, param, cons=None, c_map='GnBu_r',win_size=(1500,1000) ,cpos=None, plot_title='AEM Data',save_name = None,clip=None, ps=5,ac=None,bc=None):
        """ Plots provided data in 3D using pyvista.
        Args:
            node_list (list)    : List of node objects.
            parameter (str)     : Dictionary key for node object params attribute.

            cons (list, optional)   : List of Polydata objects. Defaults to None.
            meshes (list, optional) : List of Polydata objects. Defaults to None.

        Raises:
            Window: This function raises a visualisation window.

        """
        # variable input formats
        if isinstance(data,list): 
            core_node_list=data   


        sargs = dict(height=0.1,width=0.50, vertical=False, position_x=0.25, position_y=0.1,title=plot_title)
        kw_args= {
            'clim':clip,'below_color':bc, 'above_color':ac,
            'render_points_as_spheres':True, 'cmap':c_map,'point_size':ps
        }
        plotter = pv.Plotter(notebook=False,title=plot_title)
        # generate our plotter
        if save_name!=None:
            plotter.off_screen = True

        

        plotter.store_image = True
        plotter.window_size = win_size
        #plotter.camera_position = cpos

        pyv_list = []

        x = [node.nloc[0] for node in core_node_list]                     # 0.0 - collect the x of the points
        y = [node.nloc[1] for node in core_node_list]                     # 0.1 - collect the y of the points
        z = [node.nloc[2] for node in core_node_list]                    # 0.2 - collect the z of the points
        p = [node.params[param] for node in core_node_list]
        if param=='labl':
            boring_cmap = plt.cm.get_cmap("GnBu_r", 2)
            kw_args['cmap']=boring_cmap
            annotations = {
                0: "Cover",
                1: "Basement",
            }
            sargs['n_labels'] =0
            kw_args['annotations']=annotations

        # wow a pyvista object
        vert = list(zip(x,y,z))
        pv_cloud = pv.PolyData(vert)
        pv_cloud[param] = p         
        # lets add it to the list 
        pyv_list.append((pv_cloud,False))    

        if isinstance(cons,pd.DataFrame):
            # oh you wanna do some edges?
            points  = [x.nloc for x in core_node_list]
            edges   = list(zip(list(cons['U'].values),list(cons['V'].values)))
            edges   = np.hstack([[2,x[0],x[1]] for x in edges])
            edges   = pv.PolyData(points,lines=edges,n_lines=len(cons['U'].values))
            edges[param] = list(range(len(points)))  
            # lets go
            pyv_list.append((edges,True))
    
        # plotting done here
        for i,(pyvo,plot_e) in enumerate(pyv_list):

            if plot_e:
                plotter.add_mesh(pyvo,scalar_bar_args=sargs,show_edges=True,edge_color=[0,0,0],color='red',opacity=0.2,** kw_args,)
            if not plot_e:
                plotter.add_mesh(pyvo,scalar_bar_args=sargs,** kw_args)

            if i==0:
                pass
            #plotter.camera_set = True
        if cpos!= None:
            # visualise and save
            cpos = plotter.show(cpos=cpos)
        
        if cpos==None:
            # visualise and save
            cpos = plotter.show()
            #return(cpos)
        
        if save_name != None:
            image = plotter.image
            fig, ax = plt.subplots(figsize=(win_size[0]/100, win_size[1]/100))
            ax.imshow(image)
            plt.tight_layout()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.savefig(save_name,dpi=600)