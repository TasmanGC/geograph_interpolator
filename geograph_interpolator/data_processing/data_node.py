class CoreNode():
    
    def __init__(self,n_iD,nloc,params={}):
        ''' A Base Data Structure for the nodes of our graph, mainly used for visualisation and graph construction.'''
        self.n_iD = n_iD
        self.nloc = nloc
        self.params = params