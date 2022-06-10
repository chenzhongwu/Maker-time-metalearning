import torch.nn as nn
import torch
import dgl.function as fn
import torch.nn.functional as F


class ExtGNNLayer(nn.Module):
    def __init__(self, args, act=None):
        super(ExtGNNLayer, self).__init__()
        self.args = args
        self.act = act

        # define in/out/loop transform layer
        self.W_O = nn.Linear(args.rel_dim + args.ent_dim + args.time_dim, args.ent_dim)
        self.W_I = nn.Linear(args.rel_dim + args.ent_dim + args.time_dim, args.ent_dim)
        self.W_S = nn.Linear(args.ent_dim, args.ent_dim)

        # define relation transform layer
        self.W_R = nn.Linear(args.rel_dim, args.rel_dim)
        # 因为公式7的修改方式是相加，所以两者维度要相同
        self.W_T = nn.Linear(args.time_dim, args.time_dim)

    def msg_func(self, edges):
        # print(edges.data['h'].size())
        # print(edges.src['h'].size())
        # print(edges.data['time'].size())

        comp_h = torch.cat((edges.data['h'], edges.src['h'], edges.data['time']), dim=-1)
        # print(comp_h.size())
        # print("8978968576")
        # print(comp_h)
        non_inv_idx = (edges.data['inv'] == 0)
        inv_idx = (edges.data['inv'] == 1)
        # 公式5
        msg = torch.zeros_like(edges.src['h'])
        # print(comp_h)
        # print("000000000000000000")
        # print(self.W_I)
        # print(comp_h[non_inv_idx].size())
        msg[non_inv_idx] = self.W_I(comp_h[non_inv_idx])
        msg[inv_idx] = self.W_O(comp_h[inv_idx])

        return {'msg': msg}

    def apply_node_func(self, nodes):
        h_new = self.W_S(nodes.data['h']) + nodes.data['h_agg']

        if self.act is not None:
            h_new = self.act(h_new)

        return {'h': h_new}

    def edge_update(self, rel_emb, time_emb):
        h_edge_new = self.W_R(rel_emb)
        # print(time_emb.size())
        time_edge_new = self.W_T(time_emb)

        if self.act is not None:
            h_edge_new = self.act(h_edge_new)
            time_edge_new = self.act(time_edge_new)

        return h_edge_new, time_edge_new

    def forward(self, g,ent_emb, rel_emb, time_emb):
        with g.local_scope():
            g.edata['h'] = rel_emb[g.edata['b_rel']]
            # g.edata['h'] = rel_emb[g.edata['rel']]
            # print(time_pattern_g.ndata)
            # g.edata['time'] = time_emb[time_pattern_g.ndata['ori_idx'][g.edata['time']]]
            # print("7474744")
            # print(g.edata['time'])
            # print(time_emb)
            g.edata['time'] = time_emb[g.edata['time']]
            # g.edata['time'] = time_emb[g.edata['b_rel']]
            g.ndata['h'] = ent_emb

            g.update_all(self.msg_func, fn.mean('msg', 'h_agg'), self.apply_node_func)

            rel_emb, time_emb = self.edge_update(rel_emb, time_emb)
            ent_emb = g.ndata['h']
        # print("9090")
        # print(time_emb.size())
        return ent_emb, rel_emb, time_emb


class ExtGNN(nn.Module):
    # knowledge extrapolation with GNN
    def __init__(self, args):
        super(ExtGNN, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()

        for idx in range(args.num_layers):
            if idx == args.num_layers - 1:
                self.layers.append(ExtGNNLayer(args, act=None))
            else:
                self.layers.append(ExtGNNLayer(args, act=F.relu))

    def forward(self, g, **param):
        rel_emb = param['rel_feat']
        ent_emb = param['ent_feat']
        time_emb = param['time_feat']
        for layer in self.layers:
            ent_emb, rel_emb, time_emb  = layer(g, ent_emb, rel_emb, time_emb)

        return ent_emb, rel_emb, time_emb
