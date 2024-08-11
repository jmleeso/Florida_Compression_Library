import networkx as nx
import numpy as np

def image_to_graph(shape, k_size = 3, self_connect = False):
    # Create a graph
    G = nx.Graph()
    
    k_size//2
    connect_comp = [i for i in range(-k_size//2+1, k_size//2+1)]

    # Add nodes
    num_rows, num_cols = shape
    for i in range(num_rows):
        for j in range(num_cols):
            G.add_node((i, j))

    # Add edges
    for i in range(num_rows):
        for j in range(num_cols):
            # Check neighboring pixels
            for dx in connect_comp:
                for dy in connect_comp:
                    if dx == 0 and dy == 0:
                        if self_connect:
                            G.add_edge((i, j), (i, j))
                        else:
                            continue
                    new_x = i + dx
                    new_y = j + dy
                    if 0 <= new_x < num_rows and 0 <= new_y < num_cols:
                        G.add_edge((i, j), (new_x, new_y))
                        # G.add_edge((new_x, new_y),(i, j))

    # Generate adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    return adj_matrix

def convert_graph_index(G, img_shape):
    rows, cols = img_shape
    node_index = []
    for node in G.nodes():
        node_index.append(node[0]*cols +node[1])
        
    edge_index = []
    for edge in G.edges():
        idx_a, idx_b = edge[0][0]*cols +edge[0][1], edge[1][0]*cols +edge[1][1]
        edge_index.append((idx_a, idx_b))
        edge_index.append((idx_b, idx_a))
    
    edge_index = sorted(edge_index)
    return node_index, edge_index

def normalize_adj_matrix(adj_mat):
    D = np.diag(np.sqrt(1/np.sum(adj_mat, axis = 1)))
    return D@adj_mat@D

def tensor_to_graph(shape,  k_size = 3,  fc_connect = False, self_connect = False):
    # Create a graph
    G = nx.Graph()

    # Get the shape of the tensor
    depth, height, width = shape

    # Add nodes
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                G.add_node((d, h, w))

    # Add edges

    connect_comp = [i for i in range(-k_size//2+1, k_size//2+1)]
    if fc_connect==True:
        d_connect_comp = np.arange(depth) 
    elif type(fc_connect) == list:
        d_connect_comp = fc_connect
    elif fc_connect==False:
        d_connect_comp = connect_comp
    else:
        assert(False)
    
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                # Check neighboring voxels
                for cur_dd in d_connect_comp:
                    for dh in connect_comp:
                        for dw in connect_comp:
                            
                            dd = cur_dd - d if fc_connect==True else cur_dd
                                
                            if dd == 0 and dh == 0 and dw == 0:
                                if self_connect:
                                    G.add_edge((d, h, w), (d, h, w))
                                else:
                                    continue
                                    
                            new_d = d + dd    
                            new_h = h + dh
                            new_w = w + dw
                            
                            if 0 <= new_d < depth and 0 <= new_h < height and 0 <= new_w < width:
                                G.add_edge((d, h, w), (new_d, new_h, new_w))

    adj_matrix = nx.adjacency_matrix(G).todense()

    return adj_matrix
    
def generate_graph(dataset = "XGC", shape = [39,39], k_size=3 , self_connect = True, input_size = 1536):
    
    if dataset =="XGC":
        adj_mat = image_to_graph(shape, k_size, self_connect)
    elif dataset == "E3SM":
        adj_mat = tensor_to_graph(shape,  k_size = k_size,  fc_connect = False, self_connect = True)
    elif dataset == "S3D":
        adj_mat = tensor_to_graph(shape,  k_size = k_size,  fc_connect = [0], self_connect = True)
    
    d_size = adj_mat.shape[0]
    if d_size<=input_size:
        
        unit_mat = np.identity(input_size)
        unit_mat[:d_size,:d_size] = adj_mat
        adj_mat = unit_mat
    
    adj_mat = normalize_adj_matrix(adj_mat)
    
    return adj_mat





import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, init = True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

        # Initialize parameters
        if init:
            nn.init.xavier_uniform_(self.weight.data)
            nn.init.zeros_(self.bias.data)

    def forward(self, x, adj_matrix):
        # Input: (batch_size, num_nodes, input_dim)
        # Adjacency matrix: (batch_size, num_nodes, num_nodes)

        # Perform graph convolution
        support = torch.matmul(x, self.weight)  # (batch_size, num_nodes, output_dim)
        output = torch.matmul(adj_matrix, support)  # (batch_size, num_nodes, output_dim)
        output = output + self.bias  # Add bias

        return output
    
    
class image_GCN(nn.Module):
    def __init__(self, n_nodes= 1536, dims = [16, 32],  latent = 100, adj_size = 3, activation = F.leaky_relu, device = "cpu"):
        super(image_GCN, self).__init__()
        
        # adj_matrix = image_to_graph(img_size, k_size = adj_size)
        # print(adj_matrix)
        # self.adj_matrix = torch.FloatTensor(normalize_adj_matrix(adj_matrix))
        self.act = activation
        if device =="cuda":
            self.adj_matrix = self.adj_matrix.cuda()
            
        
        self.encoder_gc1 = GraphConvolution(1,dims[0])
        self.encoder_gc2 = GraphConvolution(dims[0],dims[1])                 
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),                                # Flatten the output,
            nn.Linear(dims[1]* n_nodes, latent),
        )
        
        self.fc2 = nn.Sequential(                              # unFlatten the output,
            nn.Linear(latent, dims[1]* n_nodes),
            nn.Unflatten(1, (n_nodes, dims[1]))
        )
        
        self.decoder_gc1 = GraphConvolution(dims[1],dims[0])
        self.decoder_gc2 = GraphConvolution(dims[0],1)

    def forward(self, x, adj_mat = None):
        if adj_mat is None:
            adj_mat = self.adj_matrix
            
        x = self.encoder_gc1(x, adj_mat)
        x = self.act(x)
        
        x = self.encoder_gc2(x, adj_mat)
        x = self.act(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.act(x)
        x = self.decoder_gc1(x, adj_mat)
        x = self.act(x)
        
        x = self.decoder_gc2(x, adj_mat)
    

        return x
    
    