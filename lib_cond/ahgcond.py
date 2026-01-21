import faiss
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import Counter
from lib_utils.metrics import evaluate
from lib_utils.cond_utils import flatten_params, index2mask
from lib_models.HNN.utils import create_coo_from_edge_index, ssm2tst
from torch_geometric.data import Data
from itertools import product
from lib_cond.ahgcond_utils import MLPAdapter, HyperConv
from lib_utils.exp_agent import parse_model


def build_syn_class_indices(num_class_dict):
    syn_class_indices = []
    s = 0
    for c in range(len(num_class_dict)):
        e = s + int(num_class_dict[c])
        syn_class_indices.append((s, e))
        s = e
    return syn_class_indices


def build_train_class_indices(labels: torch.Tensor, idx_train: torch.Tensor, nclass: int, device):
    idx_train = torch.as_tensor(idx_train, dtype=torch.long, device=device)
    class_lists = []
    y_tr = labels[idx_train]
    for c in range(nclass):
        local = torch.where(y_tr == c)[0]
        class_lists.append(idx_train[local])
    return class_lists


class AHGCondBase:
    def __init__(self, data, masks, args):
        self.data = data
        self.args = args
        self.device = args.device

        self.labels = data.y
        self.idx_train = index2mask(masks["train"]).cpu()
        self.labels_train = data.y[self.idx_train].cpu()
        self.nclass = data.y.max().cpu().numpy() + 1
        self.masks = masks

        self.generate_labels_syn()
        self.init_syn_data()
        self.reset_parameters()

    def generate_labels_syn(self):
        counter = Counter(self.labels_train.cpu().numpy())
        num_class_dict = {}

        sorted_counter = sorted(counter.items(), key=lambda x: x[0])
        labels_syn = []

        for _, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = math.ceil(num * self.args.reduction_rate)
            labels_syn += [c] * num_class_dict[c]

        self.labels_syn, _ = torch.sort(torch.LongTensor(labels_syn).to(self.device))
        self.num_class_dict = num_class_dict

        self.syn_class_indices = build_syn_class_indices(self.num_class_dict)
        self.train_class_indices = build_train_class_indices(
            self.labels, self.idx_train, self.nclass, self.device
        )

    def reset_parameters(self):
        if self.args.ac_prop == "diff":
            self.H = self.diffusion(self.data)
        elif self.args.ac_prop == "truc_pos":
            self.H = self.trunc_poisson_diff(self.data)
        elif self.args.ac_prop == "vanilla":
            self.H = self.vanilla_diff(self.data)

        if self.args.ac_ini == "random":
            X_ini = self.random_selection(self.H)
        elif self.args.ac_ini == "kmeans":
            X_ini = self.kmeans_selection(self.H)
        elif self.args.ac_ini == "random_aggr":
            X_ini = self.random_aggr_selection(self.H, self.args.aggr_M)

        self.feat_syn.data.copy_(X_ini)

    def init_syn_data(self):
        nnodes_syn = len(self.labels_syn)
        self.n = nnodes_syn
        self.e = nnodes_syn
        self.d = self.data.x.shape[1]

        self.feat_syn = nn.Parameter(torch.empty(self.n, self.d).to(self.device))
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=self.args.lr_x)

        self.MLP_phi_model = MLPAdapter(
            in_dim=self.d * 2,
            out_dim=1,
            hidden=self.args.hidden_phi,
            nlayers=self.args.nlayers_phi,
            dropout=self.args.dropout_phi,
            dropout_input=self.args.dropout_input_phi,
            with_bn=self.args.bn_phi,
            residual_ratio=self.args.residual_ratio_phi,
            with_inout_ln=self.args.inout_ln_phi,
            args=self.args,
        ).to(self.args.device)

        self.optimizer_phi = torch.optim.Adam(self.MLP_phi_model.parameters(), lr=self.args.lr_phi)

        self.iden = torch.nonzero(torch.eye(self.n, device=self.device)).T

        self.adapt_eps = nn.Parameter(torch.FloatTensor(self.n).to(self.device))
        self.optimizer_eps = torch.optim.Adam([self.adapt_eps], lr=self.args.lr_eps)
        self.eps_bound = 1e-3

        if self.args.extend_self_loop:
            self.self_loop = torch.eye(self.n, device=self.device, dtype=torch.bool)
            self.self_loop_weight = torch.eye(self.n, device=self.device, dtype=torch.float32)
            self.self_loop.requires_grad = False
            self.self_loop_weight.requires_grad = False

        print(f"Shape of the condensed hypergraph: {self.n} x {self.e}")

    def weighted_ce_graph(self, data, args):
        H = create_coo_from_edge_index(data.hyperedge_index)

        colsum = np.array(H.sum(0), dtype="float")
        r_inv_sqrt = np.power(colsum, -1).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
        B = sp.diags(r_inv_sqrt)

        A_beta = H.dot(B).dot(H.transpose())
        I = sp.eye(A_beta.shape[0])
        A_beta += I

        rowsum = np.array(A_beta.sum(1), dtype="float")
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
        D = sp.diags(r_inv_sqrt)

        A_beta = D.dot(A_beta).dot(D)

        return ssm2tst(A_beta).to(args.device)

    def diffusion(self, data):
        X = data.x
        norm_W = self.weighted_ce_graph(data, self.args)

        assert norm_W.layout == torch.sparse_coo
        norm_W = norm_W.coalesce()

        X_out = self.args.alpha_x * X
        W_power_X = X.clone()

        for l in range(1, self.args.L):
            W_power_X = torch.sparse.mm(norm_W, W_power_X)
            X_out += self.args.alpha_x * ((1 - self.args.alpha_x) ** l) * W_power_X

        W_L_X = torch.sparse.mm(norm_W, W_power_X)
        X_out += ((1 - self.args.alpha_x) ** self.args.L) * W_L_X

        return X_out

    def trunc_poisson_diff(self, data):
        lam = self.args.lam
        normalize_weights = self.args.normalize_weights

        X = data.x
        norm_W = self.weighted_ce_graph(data, self.args)

        assert norm_W.layout == torch.sparse_coo
        norm_W = norm_W.coalesce()

        w = math.exp(-lam)
        X_out = w * X
        w_sum = w if normalize_weights else 1.0

        W_power_X = X

        K = getattr(self.args, "K", None)
        if K is None:
            K = int(lam + 3.0 * math.sqrt(max(lam, 1e-8))) + 1

        for l in range(1, K + 1):
            W_power_X = torch.sparse.mm(norm_W, W_power_X)
            w = w * (lam / l)
            X_out = X_out + w * W_power_X
            if normalize_weights:
                w_sum += w

        if normalize_weights:
            X_out = X_out / (w_sum + 1e-8)

        return X_out

    def vanilla_diff(self, data):
        conv = HyperConv(self.d, self.d)
        X_out = data.x
        for _ in range(self.args.L):
            X_out = conv(X_out, hyperedge_index=data.hyperedge_index)
        return X_out

    def random_selection(self, X):
        feat_syn = []
        for c in range(self.nclass):
            lab_idx = torch.where(self.labels_train == c)[0]
            x_idx = self.idx_train[lab_idx]
            x_mhp_c = X[x_idx]

            k = self.num_class_dict[c]
            indices = torch.randperm(x_mhp_c.size(0))[:k]
            selected_rows = x_mhp_c[indices]
            feat_syn.append(selected_rows)

        feat_syn = torch.cat(feat_syn, dim=0)
        return feat_syn.to(self.device)

    def random_aggr_selection(self, X, M=10):
        idx_train = torch.as_tensor(self.idx_train, dtype=torch.long, device=self.device)

        feat_syn = torch.zeros((self.n, self.d), device=self.device, dtype=X.dtype)
        sampled_idxes = [[] for _ in range(self.n)]

        for c in range(self.nclass):
            start, end = self.syn_class_indices[c]
            n_syn_c = end - start

            idx_c = idx_train[self.data.y[idx_train] == c]
            if idx_c.numel() == 0:
                continue

            idx_c = idx_c[torch.randperm(idx_c.numel(), device=self.device)]
            l = idx_c.numel()

            for i in range(n_syn_c):
                syn_i = start + i
                pos = torch.arange(i * M, (i + 1) * M, device=self.device) % l
                sampled = idx_c[pos]

                sampled_idxes[syn_i] = sampled.tolist()
                feat_syn[syn_i] = X[sampled].mean(dim=0)

        return feat_syn

    def kmeans_selection(self, X, use_gpu=True, niter=10, nredo=1, seed=0, normalize=False):
        X = X.to(self.device)
        feat_syn = []
        d = int(X.size(1))

        if use_gpu:
            try:
                ng = faiss.get_num_gpus()
                if ng <= 0:
                    use_gpu = False
            except Exception:
                use_gpu = False

        for c in range(self.nclass):
            lab_idx = torch.where(self.labels_train == c)[0]
            x_idx = self.idx_train[lab_idx]
            x_mhp_c = X[x_idx]

            k = int(self.num_class_dict[c])
            n_c = int(x_mhp_c.size(0))

            if n_c == 0 or k <= 0:
                continue

            if n_c <= k:
                feat_syn.append(x_mhp_c)
                continue

            x_np = x_mhp_c.detach().float().cpu().numpy().astype("float32")

            if normalize:
                faiss.normalize_L2(x_np)

            kmeans = faiss.Kmeans(
                d=d,
                k=k,
                niter=niter,
                nredo=nredo,
                seed=seed,
                gpu=use_gpu,
                verbose=False,
            )
            kmeans.cp.min_points_per_centroid = 1
            kmeans.train(x_np)
            centers_np = kmeans.centroids

            if normalize:
                faiss.normalize_L2(centers_np)

            centers = torch.from_numpy(centers_np).to(self.device).type_as(X)
            feat_syn.append(centers)

        return torch.cat(feat_syn, dim=0)

    def multi_hop_embed_and_aggr(self, aggr_mode="mean"):
        if self.args.align_mode == "diff":
            embed_ori = self.diffusion(self.data)
        elif self.args.align_mode == "truc_pos":
            embed_ori = self.trunc_poisson_diff(self.data)
        elif self.args.align_mode == "vanilla":
            embed_ori = self.vanilla_diff(self.data)

        ori_cls_prototype = []
        self.coeff = []
        self.coeff_sum = 0

        for c in range(self.nclass):
            if c in self.num_class_dict:
                index = torch.where(self.labels_train == c)
                coe = self.num_class_dict[c] / max(self.num_class_dict.values())
                self.coeff_sum += coe
                self.coeff.append(coe)
                if aggr_mode == "mean":
                    ori_cls_prototype.append(embed_ori[index].mean(dim=0).to(self.device))
                elif aggr_mode == "sum":
                    ori_cls_prototype.append(embed_ori[index].sum(dim=0).to(self.device))

        self.cls_embed_real = torch.stack(ori_cls_prototype, dim=0)
        self.coeff_sum = torch.tensor(self.coeff_sum).to(self.device)

    def cls_disc_loss(self, embed_syn):
        if not hasattr(self, "cls_embed_real") or self.cls_embed_real is None:
            self.multi_hop_embed_and_aggr(self.args.cls_aggr)
            self.cls_embed_real = F.normalize(input=self.cls_embed_real, p=2, dim=1)

        syn_cls_prototype = []

        for c in range(self.nclass):
            if c in self.num_class_dict:
                index = torch.where(self.labels_syn == c)
                if self.args.cls_aggr == "mean":
                    syn_cls_prototype.append(embed_syn[index].mean(dim=0).to(self.device))
                elif self.args.cls_aggr == "sum":
                    syn_cls_prototype.append(embed_syn[index].sum(dim=0).to(self.device))

        cls_embed_syn = torch.stack(syn_cls_prototype, dim=0)
        cls_embed_syn = F.normalize(input=cls_embed_syn, p=2, dim=1)

        cov_embed = self.cls_embed_real @ cls_embed_syn.T
        iden = torch.eye(self.nclass).cuda()
        class_loss = F.mse_loss(cov_embed, iden)

        return class_loss

    def div_loss(self, embed_syn):
        loss = 0.0
        total = embed_syn.size(0)

        for c in range(self.nclass):
            if c in self.num_class_dict:
                index = torch.where(self.labels_syn == c)
                Xc = embed_syn[index]

                Kc = Xc.size(0)
                if Kc <= 1:
                    continue

                Xc = F.normalize(Xc, p=2, dim=1, eps=self.args.norm_eps)
                G = Xc @ Xc.t()
                off = G - torch.diag(torch.diag(G))

                loss_c = 0.5 * (off**2).sum() / (Kc * (Kc - 1))

                coeff = Kc / total
                loss += coeff * loss_c

        return loss

    def dist_match_loss(self, embed_syn):
        if not hasattr(self, "cls_embed_real") or self.cls_embed_real is None:
            self.multi_hop_embed_and_aggr(aggr_mode="mean")

        dist_loss = torch.tensor(0.0).to(self.device)
        loss_fn = nn.MSELoss()

        for c in range(self.nclass):
            if c in self.num_class_dict:
                index = torch.where(self.labels_syn == c)
                dist_loss += self.coeff[c] * loss_fn(self.cls_embed_real[c], embed_syn[index].mean(dim=0))

        dist_loss = dist_loss / (self.coeff_sum)

        return dist_loss

    def contrast_loss(self, embed_syn):
        n_neg = self.args.n_neg_adjust

        idx_train = torch.as_tensor(self.idx_train, dtype=torch.long, device=self.device)
        perm = torch.randperm(idx_train.numel(), device=self.device)
        bs = min(self.args.bs_adjust, idx_train.numel())
        anchor_idx = idx_train[perm[:bs]]

        anchor_feat = self.H[anchor_idx].to(self.device)
        anchor_labels = self.labels[anchor_idx].to(self.device)

        pos_idx = torch.empty(bs, dtype=torch.long, device=self.device)

        for c, (st, ed) in enumerate(self.syn_class_indices):
            mask = (anchor_labels == c)
            num_c = int(mask.sum().item())
            if num_c > 0:
                pos_idx[mask] = torch.randint(st, ed, (num_c,), device=self.device)

        n_syn = embed_syn.size(0)
        neg_idx = torch.randint(0, n_syn, (bs, n_neg), device=self.device)

        pos_h = embed_syn[pos_idx]
        neg_h = embed_syn[neg_idx]

        pos_logits = (anchor_feat * pos_h).sum(dim=1)
        neg_logits = (anchor_feat.unsqueeze(1) * neg_h).sum(dim=2).reshape(-1)

        loss_func = nn.BCEWithLogitsLoss(reduction="none")
        pos_losses = loss_func(pos_logits, torch.ones_like(pos_logits))
        neg_losses = loss_func(neg_logits, torch.zeros_like(neg_logits))

        contrast_loss = pos_losses.mean() + neg_losses.mean()
        return contrast_loss

    def fine_grained_supcon_loss(
        self,
        embed_syn,
        args,
        normalize: bool = True,
        detach_pool: bool = True,
        replace_if_needed: bool = True,
    ):
        n_pos, n_neg = args.n_pos_fine, args.n_neg_fine
        temp = args.temp

        device = embed_syn.device
        feat_all = self.H.to(device)

        idx_train = torch.as_tensor(self.idx_train, dtype=torch.long, device=device)

        B = self.n
        a, y_a = embed_syn, self.labels_syn

        pool = feat_all.detach() if detach_pool else feat_all

        if normalize:
            a = F.normalize(a, dim=-1)
            pool = F.normalize(pool, dim=-1)

        pos_idx = torch.empty((B, n_pos), dtype=torch.long, device=device)
        neg_idx = torch.empty((B, n_neg), dtype=torch.long, device=device)

        for i in range(B):
            c = int(y_a[i].item())

            pos_pool = self.train_class_indices[c]
            m = pos_pool.numel()
            if m == 0:
                pos_idx[i] = idx_train[torch.randint(0, idx_train.numel(), (n_pos,), device=device)]
            else:
                if m >= n_pos:
                    sel = torch.randperm(m, device=device)[:n_pos]
                    pos_idx[i] = pos_pool[sel]
                else:
                    if replace_if_needed:
                        sel = torch.randint(0, m, (n_pos,), device=device)
                        pos_idx[i] = pos_pool[sel]
                    else:
                        rep = torch.arange(n_pos, device=device) % m
                        pos_idx[i] = pos_pool[rep]

            need = n_neg
            collected = []
            while need > 0:
                cand = idx_train[torch.randint(0, idx_train.numel(), (need * 2,), device=device)]
                cand = cand[self.labels[cand] != c]
                if cand.numel() == 0:
                    continue
                take = cand[:need]
                collected.append(take)
                need -= take.numel()
            neg_idx[i] = torch.cat(collected, dim=0)[:n_neg]

        pos = pool[pos_idx]
        neg = pool[neg_idx]

        sim_pos = torch.einsum("bd,bpd->bp", a, pos) / temp
        sim_neg = torch.einsum("bd,bnd->bn", a, neg) / temp

        sim_neg = sim_neg * self.args.neg_scale

        all_logits = torch.cat([sim_pos, sim_neg], dim=1)
        log_denom = torch.logsumexp(all_logits, dim=1, keepdim=True)
        loss_per_anchor = -(sim_pos - log_denom).mean(dim=1)
        loss = loss_per_anchor.mean()

        return loss, (pos_idx, neg_idx)

    def fine_grained_supcon_loss_hard_neg(
        self,
        embed_syn,
        args,
        normalize: bool = True,
        detach_pool: bool = True,
        replace_if_needed: bool = True,
    ):
        n_pos, n_neg = args.n_pos_fine, args.n_neg_fine
        n_cand = args.n_cand_ratio * n_neg
        temp = args.temp
        neg_scale = args.neg_scale

        device = embed_syn.device
        feat_all = self.H.to(device)

        idx_train = torch.as_tensor(self.idx_train, dtype=torch.long, device=device)

        B = self.n
        a, y_a = embed_syn, self.labels_syn

        pool = feat_all.detach() if detach_pool else feat_all

        if normalize:
            a = F.normalize(a, dim=-1)
            pool = F.normalize(pool, dim=-1)

        pos_idx = torch.empty((B, n_pos), dtype=torch.long, device=device)

        for i in range(B):
            c = int(y_a[i].item())
            pos_pool = self.train_class_indices[c]
            m = pos_pool.numel()

            if m == 0:
                pos_idx[i] = idx_train[torch.randint(0, idx_train.numel(), (n_pos,), device=device)]
            else:
                if m >= n_pos:
                    sel = torch.randperm(m, device=device)[:n_pos]
                    pos_idx[i] = pos_pool[sel]
                else:
                    if replace_if_needed:
                        sel = torch.randint(0, m, (n_pos,), device=device)
                        pos_idx[i] = pos_pool[sel]
                    else:
                        rep = torch.arange(n_pos, device=device) % m
                        pos_idx[i] = pos_pool[rep]

        pos = pool[pos_idx]
        sim_pos = torch.einsum("bd,bpd->bp", a, pos) / temp

        neg = torch.empty((B, n_neg, pool.size(1)), device=device)
        sim_neg = torch.empty((B, n_neg), device=device)

        for i in range(B):
            c = int(y_a[i].item())

            neg_pool = idx_train[self.labels[idx_train] != c]
            if neg_pool.numel() == 0:
                cand_idx = idx_train[torch.randint(0, idx_train.numel(), (n_cand,), device=device)]
            else:
                cand_idx = neg_pool[torch.randint(0, neg_pool.numel(), (n_cand,), device=device)]

            cand = pool[cand_idx]
            sim_cand = torch.matmul(a[i], cand.T)

            hard_k = sim_cand.topk(n_neg, largest=True).indices
            neg[i] = cand[hard_k]
            sim_neg[i] = sim_cand[hard_k]

        sim_neg = sim_neg / temp
        sim_neg = sim_neg * neg_scale

        all_logits = torch.cat([sim_pos, sim_neg], dim=1)
        log_denom = torch.logsumexp(all_logits, dim=1, keepdim=True)
        loss_per_anchor = -(sim_pos - log_denom).mean(dim=1)
        loss = loss_per_anchor.mean()

        return loss, None

    def topo_generator(self, feat_syn):
        edge_index = np.array(list(product(range(self.n), range(self.n))))
        edge_index = edge_index.T

        node_pairs = torch.cat([feat_syn[edge_index[0]], feat_syn[edge_index[1]]], axis=1)
        H_hat = self.MLP_phi_model(node_pairs).view(self.n, self.n).T

        if self.args.filter_mode == "sigmoid":
            H_soft = torch.sigmoid((H_hat - self.adapt_eps.unsqueeze(0)) / self.args.tau_eps)
            if self.args.internal_self_loop:
                H_soft = H_soft - torch.diag(torch.diag(H_soft, 0)) + torch.diag(
                    torch.full((self.n,), self.args.loop_weight, device=self.device)
                )
            H_hard = H_soft > 0.5

        elif self.args.filter_mode == "gated":
            G = torch.sigmoid((H_hat - self.adapt_eps) / self.args.tau_eps)
            W = torch.nn.functional.softplus(H_hat)
            W = torch.clamp(W, max=self.args.w_max)

            H_soft = G * W
            H_hard = G > 0.5

        elif self.args.filter_mode == "dual_sigmoid":
            H_hat = torch.sigmoid(H_hat)
            eps = self.args.eps_bound + (1 - 2 * self.args.eps_bound) * torch.sigmoid(self.adapt_eps)

            H_soft = torch.sigmoid((H_hat - eps.unsqueeze(0)) / self.args.tau_eps)
            if self.args.internal_self_loop:
                H_soft = H_soft - torch.diag(torch.diag(H_soft, 0)) + torch.diag(
                    torch.full((self.n,), self.args.loop_weight, device=self.device)
                )
            H_hard = H_soft > 0.5

        elif self.args.filter_mode == "hard":
            H_soft = torch.sigmoid(H_hat)
            if self.args.internal_self_loop:
                H_soft = H_soft - torch.diag(torch.diag(H_soft, 0)) + torch.diag(
                    torch.full((self.n,), self.args.loop_weight, device=self.device)
                )
            eps = self.args.eps_bound + (1 - 2 * self.args.eps_bound) * torch.sigmoid(self.adapt_eps)
            H_hard = H_soft > eps.unsqueeze(0)

        if self.args.extend_self_loop:
            H_hard = torch.cat([self.self_loop, H_hard], dim=1)
            H_soft = torch.cat([self.self_loop_weight, H_soft], dim=1)

        edge_index_syn = H_hard.nonzero(as_tuple=False).t().contiguous()
        edge_weight_syn = H_soft[H_hard].contiguous()

        return edge_index_syn, edge_weight_syn

    def test_with_val(self, cond_data, data, masks, args):
        model = parse_model(self.args, data)
        model = model.to(self.device)

        criterion = nn.NLLLoss()

        start_time = time.time()
        model.reset_parameters()

        if args.method == "UniGCNII":
            optimizer = torch.optim.Adam(
                [
                    dict(params=model.reg_params, weight_decay=0.01),
                    dict(params=model.non_reg_params, weight_decay=5e-4),
                ],
                lr=0.01,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            out, _ = model(cond_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, cond_data.y)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_thresh)
            optimizer.step()

            if (epoch + 1) % args.display_step == 0:
                result = evaluate(model, data, masks)
                print(
                    f"Epoch: {epoch+1:02d}, "
                    f"Train Acc: {100 * result[0]:.2f}%, "
                    f"Valid Acc: {100 * result[1]:.2f}%, "
                    f"Test  Acc: {100 * result[2]:.2f}%"
                )

        end_time = time.time()
        print(f"Training Time: {end_time-start_time:.2f}")

        result = evaluate(model, data, masks)
        return result


class AHGCDD(AHGCondBase):
    def __init__(self, data, masks, args):
        super(AHGCDD, self).__init__(data, masks, args)

    def reduce(self, data):
        best_val = -1

        print("Starting data condensation via AHGCond...")

        for it in tqdm(range(self.args.cond_epochs)):
            self.optimizer_feat.zero_grad()
            self.optimizer_phi.zero_grad()
            self.optimizer_eps.zero_grad()

            feat_syn = self.feat_syn
            edge_index_syn, edge_weight_syn = self.topo_generator(feat_syn)

            conv = HyperConv(self.d, self.d)
            embed_syn = feat_syn
            for _ in range(self.args.prop_steps):
                embed_syn = conv(embed_syn, hyperedge_index=edge_index_syn, hypericd_weight=edge_weight_syn)

            if self.args.dt_loss == "coarse_disc":
                loss = self.cls_disc_loss(embed_syn)
            elif self.args.dt_loss == "fine_disc":
                loss, _ = self.fine_grained_supcon_loss(embed_syn, self.args)
            elif self.args.dt_loss == "dist_match":
                loss = self.dist_match_loss(embed_syn)
            elif self.args.dt_loss == "contrast":
                loss = self.contrast_loss(embed_syn)
            elif self.args.dt_loss in ["dual", "dual_contrast"]:
                alpha = math.pi * it / (2 * self.args.cond_epochs)
                w_1 = math.cos(alpha)
                w_2 = math.sin(alpha)

                loss_c = self.cls_disc_loss(embed_syn)
                if self.args.dt_loss == "dual":
                    if not self.args.is_hard_neg:
                        loss_f, _ = self.fine_grained_supcon_loss(embed_syn, self.args)
                    else:
                        loss_f, _ = self.fine_grained_supcon_loss_hard_neg(embed_syn, self.args)
                else:
                    loss_f = self.contrast_loss(embed_syn)

                if self.args.dynamic == "forward":
                    loss = w_1 * loss_c + w_2 * loss_f
                elif self.args.dynamic == "reverse":
                    loss = w_2 * loss_c + w_1 * loss_f
                elif self.args.dynamic == "vanilla":
                    loss = self.args.w_c * loss_c + self.args.w_f * loss_f

            if self.args.w_div == 0:
                div_loss = 0
            else:
                div_loss = self.div_loss(embed_syn)

            loss += div_loss
            loss.backward()

            if it % (self.args.tau_s + self.args.tau_f) < self.args.tau_s:
                self.optimizer_phi.step()
                self.optimizer_eps.step()
            else:
                self.optimizer_feat.step()

            if (it + 1) % self.args.cond_eval_step == 0:
                print(f"Training Iteratoin: {it+1}, loss: {loss.item()}")

                with torch.no_grad():
                    feat_syn = self.feat_syn.detach()
                    edge_index_syn, edge_weight_syn = self.topo_generator(feat_syn)
                    cond_data = Data(
                        x=feat_syn,
                        hyperedge_index=edge_index_syn,
                        icd_weight=edge_weight_syn,
                        y=self.labels_syn,
                    )

                result = self.test_with_val(cond_data, data, self.masks, self.args)

                if best_val < result[1]:
                    self.cond_data = cond_data
                    best_val = result[1]
                    print(f"Best Validation Acc: {best_val}")

        return self.cond_data


class AHGCDDX(AHGCondBase):
    def __init__(self, data, masks, args):
        super(AHGCDDX, self).__init__(data, masks, args)

    def reduce(self, data):
        es_flag = 0
        best_val = -1

        if self.args.cond_mode == "pre":
            with torch.no_grad():
                feat_syn = self.feat_syn.detach()
                edge_index_syn = self.iden
                self.cond_data = Data(x=feat_syn, hyperedge_index=edge_index_syn, y=self.labels_syn)

            result = self.test_with_val(self.cond_data, data, self.masks, self.args)

        else:
            print("Starting data condensation via AHGCondX...")

            for it in tqdm(range(self.args.cond_epochs)):
                self.optimizer_feat.zero_grad()
                feat_syn = self.feat_syn

                if self.args.dt_loss == "coarse_disc":
                    loss = self.cls_disc_loss(feat_syn)
                elif self.args.dt_loss == "fine_disc":
                    loss, _ = self.fine_grained_supcon_loss(feat_syn, self.args)
                elif self.args.dt_loss == "dist_match":
                    loss = self.dist_match_loss(feat_syn)
                elif self.args.dt_loss == "contrast":
                    loss = self.contrast_loss(feat_syn)
                elif self.args.dt_loss in ["dual", "dual_contrast"]:
                    alpha = math.pi * it / (2 * self.args.cond_epochs)
                    w_1 = math.cos(alpha)
                    w_2 = math.sin(alpha)

                    loss_c = self.cls_disc_loss(feat_syn)
                    if self.args.dt_loss == "dual":
                        if not self.args.is_hard_neg:
                            loss_f, _ = self.fine_grained_supcon_loss(feat_syn, self.args)
                        else:
                            loss_f, _ = self.fine_grained_supcon_loss_hard_neg(feat_syn, self.args)
                    else:
                        loss_f = self.contrast_loss(feat_syn)

                    if self.args.dynamic == "forward":
                        loss = w_1 * loss_c + w_2 * loss_f
                    elif self.args.dynamic == "reverse":
                        loss = w_2 * loss_c + w_1 * loss_f
                    elif self.args.dynamic == "vanilla":
                        loss = self.args.w_c * loss_c + self.args.w_f * loss_f

                    print(f"epoch-{it}: loss_c-{loss_c.item()}, loss_f-{loss_f.item()}")

                if self.args.w_div == 0:
                    div_loss = 0
                else:
                    div_loss = self.div_loss(feat_syn)

                loss += div_loss
                loss.backward()
                self.optimizer_feat.step()

                if (it + 1) % self.args.cond_eval_step == 0:
                    print(f"Training Iteratoin: {it+1}, loss: {loss.item()}")

                    with torch.no_grad():
                        feat_syn = self.feat_syn.detach()
                        edge_index_syn = self.iden
                        cond_data = Data(x=feat_syn, hyperedge_index=edge_index_syn, y=self.labels_syn)

                    result = self.test_with_val(cond_data, data, self.masks, self.args)

                    if best_val < result[1]:
                        self.cond_data = cond_data
                        best_val = result[1]
                        print(f"Best Validation Acc: {best_val}")
                        es_flag = 0
                    else:
                        es_flag += 1

                    if es_flag >= self.args.es_top:
                        break

        return self.cond_data
