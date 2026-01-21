import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from lib_dataset.edge_sampler import SNSSampler
import scipy.sparse as sp
from lib_models.HNN.utils import create_coo_from_edge_index,ssm2tst

def transition_matrix(data,args):

    H = create_coo_from_edge_index(data.hyperedge_index) 

    # 1. H x B x H^{T}
    colsum = np.array(H.sum(0),dtype='float')
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)

    A_beta = H.dot(B).dot(H.transpose())
    I = sp.eye(A_beta.shape[0])
    A_beta += I

    # 2. D^{-1/2} x H x B x H^{T} x D^{1/2}
    rowsum = np.array(A_beta.sum(1),dtype='float')
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)

    A_beta = D.dot(A_beta).dot(D)

    return ssm2tst(A_beta).to(args.device)

def index2mask(mask):
    idx = torch.nonzero(mask, as_tuple=True)[0]  
    return idx 

def extract_subhypergraph(data,mask_id):

    hyperedge_index = data.hyperedge_index.to('cpu')

    subset_ids=torch.isin(hyperedge_index[0],mask_id)
    sub_hypergraph=hyperedge_index[:,subset_ids]

    relabel_map=dict(zip(mask_id.tolist(), [i for i in range(len(mask_id))]))
    relabel_src_idx=torch.tensor([relabel_map[v.item()] for v in sub_hypergraph[0]])
    _, relabel_dst_idx = torch.unique(sub_hypergraph[1], return_inverse=True)

    sub_hyperedge_index=torch.stack((relabel_src_idx,relabel_dst_idx),dim=0)

    return sub_hyperedge_index

# =================== Negative Link Sampling ==================#

def neg_link_sampling(pos_edge_list,args):
    
    neg_edge_list = sns_negative_sampling(pos_edge_list)

    neg_hyperedge_index = convert_edge_list_to_hyperedge_index(neg_edge_list,args.device)

    return neg_hyperedge_index

def sns_negative_sampling(pos_edge_list):
    HE = [frozenset(nodes) for nodes in pos_edge_list]
    sns = SNSSampler(len(HE))
    t_sns = sns(set(HE))
    t_sns = [list(edge) for edge in t_sns]
    return t_sns

def build_pos_edge_list(hyperedge_index):
    """
    hyperedge_index: LongTensor of shape (2, num_edges)
    returns: list of LongTensors, pos_edge_list[e] contains node indices in hyperedge e
    """
    src, dst = hyperedge_index  # src: node ids, dst: hyperedge ids
    E = int(dst.max().item()) + 1  # total number of hyperedges
    pos_edge_list = [[] for _ in range(E)]

    for node_id, edge_id in zip(src.tolist(), dst.tolist()):
        pos_edge_list[edge_id].append(node_id)

    # Convert each to a LongTensor
    pos_edge_list = [torch.tensor(edge, dtype=torch.long) for edge in pos_edge_list]
    return pos_edge_list

def convert_edge_list_to_hyperedge_index(edge_list,device):

    src = []
    dst = []
    
    for hyperedge_id, nodes in enumerate(edge_list):
        src.extend(nodes)  
        dst.extend([hyperedge_id] * len(nodes)) 
    
    hyperedge_index = torch.tensor([src, dst], dtype=torch.long)
    return hyperedge_index.to(device)

# ============= Gradient Matching ==================#
def reshape_gw(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])

    return gwr, gws

def distance_wb(args, gwr, gws):
    shape = gwr.shape
    gwr, gws = reshape_gw(gwr, gws)

    if len(shape) == 1:
        return 0
    
    if args.dis_metric == "ctrl":
        alpha = 1 - args.beta
        beta = args.beta

        cosine_similarity = F.cosine_similarity(gwr, gws, dim=-1)
        euclidean_distance = torch.norm(gwr - gws, dim=-1)

        distance = alpha * (1 - cosine_similarity) + beta * euclidean_distance
    else:
        distance = 1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    return torch.sum(distance)


def grad_match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)
    torch.autograd.set_detect_anomaly(True)
    if args.dis_metric == "ours" or args.dis_metric == "ctrl":

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(args, gwr, gws)

    elif args.dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    
    elif args.dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        exit("DC error: unknown distance function")

    return dis

# ============= Parameter Matching ==================#

def flatten_params(params):
    return torch.cat([p.view(-1) for p in params])

def param_match_loss(theta_t_S, theta_t_T, theta_0_T, eps=1e-8):
    numerator = torch.norm(theta_t_S - theta_t_T, p=2) ** 2
    denominator = torch.norm(theta_0_T - theta_t_T, p=2) ** 2 + eps  
    return numerator / denominator

# =================== Batch Training ==================#

def sparse_edge_tensor(hyperedge_index):
    
    H=SparseTensor(row=hyperedge_index[0],col=hyperedge_index[1]) # |V|x|E|
    H=H.fill_value(1.) 
    
    A=H.matmul(H.t()) 
    row,col,edge_attr=A.coo()
    edge_index=torch.stack([row,col])
    
    return edge_index,edge_attr

def convert_batch_hypergraph(data,batch,device):

    batch_mask=torch.isin(data.hyperedge_index[0],batch.n_id)
    sub_hypergraph=data.hyperedge_index[:,batch_mask]

    relabel_map=dict(zip(batch.n_id.tolist(), [i for i in range(len(batch.n_id))]))
    relabel_src_idx=torch.tensor([relabel_map[v.item()] for v in sub_hypergraph[0]]).to(device)
    _, relabel_dst_idx = torch.unique(sub_hypergraph[1], return_inverse=True)

    batch_edge_index=torch.stack((relabel_src_idx,relabel_dst_idx),dim=0)

    return Data(x=batch.x,hyperedge_index=batch_edge_index)