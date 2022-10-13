import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv


from .gnn_layers import WGraphConv
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feat , out_feats, num_hlayers=0):
        super(GCN, self).__init__()
        # first layer
        self.layers = nn.ModuleList([GraphConv(in_feats, hid_feat)])

        # hidden layers
        if num_hlayers > 0:
            self.layers.extend([GraphConv(hid_feat,hid_feat) for i in range(num_hlayers)])
        
        # output layer
        self.layers.extend([GraphConv(hid_feat, out_feats)])

    def forward(self, g, in_feat):
        attention_list = []
        
        for i in range(len(self.layers)):
            if i==0:
                h = self.layers[i](g, in_feat)
            if i!=0:
                h = self.layers[i](g, h)
            
            if i!=len(self.layers):
                h = F.relu(h)

        h = F.softmax(h,dim=1)
        return(h)


class wGCN(nn.Module):
    def __init__(self, in_feats, hid_feat , out_feats, num_hlayers=0):
        super(wGCN, self).__init__()
        # first layer
        self.layers = nn.ModuleList([WGraphConv(in_feats, hid_feat)])

        # hidden layers
        if num_hlayers > 0:
            self.layers.extend([WGraphConv(hid_feat,hid_feat) for i in range(num_hlayers)])
        
        # output layer
        self.layers.extend([WGraphConv(hid_feat, out_feats)])

    def forward(self, g, in_feat):
        
        for i in range(len(self.layers)):
            if i==0:
                h = self.layers[i](g, in_feat,g.edata['W'])
            if i!=0:
                h = self.layers[i](g, h, g.edata['W'])

            if i!=len(self.layers):
                h = F.relu(h)
      
        h = F.softmax(h,dim=1)
        return(h)

class GAT(nn.Module):
    def __init__(self, in_feats, hid_feat , out_feats, num_heads, num_hlayers=0):
        super(GAT, self).__init__()
        # first layer
        self.layers = nn.ModuleList([GATConv(in_feats, hid_feat, num_heads)])

        # hidden layers
        if num_hlayers > 0:
            self.layers.extend([GATConv(hid_feat*num_heads,hid_feat,num_heads) for i in range(num_hlayers)])
        
        # output layer
        self.layers.extend([GATConv(hid_feat*num_heads, out_feats, 1)])

    def forward(self, g, in_feat):
        
        
        for i in range(len(self.layers)):
            if i==0:
                h = self.layers[i](g, in_feat, get_attention=False)
            if i!=0:
                h = self.layers[i](g, h, get_attention=False)
                # Concat last 2 dim (num_heads * out_dim)
            h = h.view(-1, h.size(1) * h.size(2)) # (in_feat, num_heads, out_dim) -> (in_feat, num_heads * out_dim)
            if i!=len(self.layers):
                h = F.relu(h)
            # append attention_list

        # Sueeze the head dim as it's = 1 
        h = h.squeeze() # (in_feat, 1, out_dim) -> (in_feat, out_dim)
        h = F.softmax(h,dim=1)
        return(h)