import argparse
import os
import yaml
from lib_dataset import _single_datasets_, _multi_datasets_


def update_from_dict(obj, updates):
    for key, value in updates.items():
        if key in ["init"] and obj.init is not None:
            continue
        setattr(obj, key, value)


def method_config(args):
    """
    is_default:
      True: use default config
      False: use dataset-specific config
    """
    if args.is_default:
        config_name = "default"
    else:
        config_name = args.dname

    try:
        task_prefix = args.task_type.split("_")[0] + "_yamls"
        conf_dt = yaml.safe_load(
            open(f"{os.path.join('./', 'lib_yamls', task_prefix, 'config_' + args.method.lower())}.yaml")
        )[config_name]
        update_from_dict(args, conf_dt)
    except Exception:
        print("No config file found or error in config format, please use method_config(args)")

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_task_args(args):
    if args.task_type == "node_cls":
        if args.dname not in _single_datasets_:
            raise ValueError("The dataset is not suitable for node classification")
        args.add_self_loop = True
        args.train_prop, args.valid_prop = 0.5, 0.25
    elif args.task_type == "hg_cls":
        if args.dname not in _multi_datasets_:
            raise ValueError("The datasets is not suitable for hypergraph classification")
        args.add_self_loop = False
        args.train_prop, args.valid_prop = 0.8, 0.1
        if args.method == "EHNN":
            raise ValueError(f"{args.method} is not supported for hypergraph classification task")
    else:
        if args.dname not in _single_datasets_:
            raise ValueError("The dataset is not suitable for edge prediction")
        args.add_self_loop = False
        args.train_prop, args.valid_prop = 0.6, 0.2

    return args


def parameter_parser():
    """
    Parse command line parameters.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_prop", type=float, default=0.5)
    parser.add_argument("--valid_prop", type=float, default=0.25)

    parser.add_argument("--cond_method", default="ahgcdd")
    parser.add_argument("--cond_epochs", default=100, type=int)
    parser.add_argument(
        "--dname",
        default="yelp",
        choices=[
            "cora",
            "pubmed",
            "coauthor_dblp",
            "yelp",
            "walmart-trips-100",
            "magpm_mini"
        ],
    )

    parser.add_argument("--task_type", default="node_cls", choices=["node_cls", "edge_pred", "hg_cls"])
    parser.add_argument("--is_default", default=False, help="Use default config or dataset-specific config")
    parser.add_argument("--use_processed", default=True)
    parser.add_argument("--method", default="HGNN")
    parser.add_argument("--device", default=0, choices=[-1, 0, 1])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--clip_grad", default=False, type=bool)
    parser.add_argument("--clip_thresh", default=5.0, type=float)
    parser.add_argument("--num_splits", type=int, default=10, help="Number of splits for self-supervised setting")
    parser.add_argument("--display_step", type=int, default=50)

    parser.add_argument("--embedding_mode", default=False, type=bool)
    parser.add_argument("--embedding_hidden", default=128, type=int)

    parser.add_argument("--normtype", default="all_one")
    parser.add_argument("--add_self_loop", action="store_false")
    parser.add_argument("--exclude_self", action="store_true")

    parser.add_argument("--feature_noise", default="0.6", type=str)

    parser.add_argument("--reduction_rate", default=0.005)
    parser.add_argument("--cond_model", default="HGNN")

    parser.add_argument("--cond_mode", default="learn", choices=["pre", "learn"])
    parser.add_argument("--ac_ini", default="random_aggr", choices=["random", "kmeans", "random_aggr"])
    parser.add_argument("--ac_prop", default="truc_pos", choices=["diff", "truc_pos", "vanilla"])
    parser.add_argument("--align_mode", default="truc_pos", choices=["diff", "truc_pos", "vanilla"])
    parser.add_argument("--alpha_x", default=0.2)
    parser.add_argument("--L", default=2)
    parser.add_argument("--lam", type=int, default=2)
    parser.add_argument("--aggr_M", type=int, default=10)
    parser.add_argument("--K", default=None)
    parser.add_argument("--normalize_weights", type=str2bool, default=False)

    parser.add_argument("--nlayers_phi", type=int, default=2)
    parser.add_argument("--hidden_phi", type=int, default=128)
    parser.add_argument("--dropout_phi", type=float, default=0.5)
    parser.add_argument("--dropout_input_phi", type=float, default=0.0)
    parser.add_argument("--inout_ln_phi", default=0)
    parser.add_argument("--bn_phi", type=int, default=1)
    parser.add_argument("--residual_ratio_phi", type=float, default=0.0, choices=[0.0, 0.25, 0.50, 0.75])

    parser.add_argument("--norm_eps", type=float, default=1e-6)
    parser.add_argument("--tau_eps", type=float, default=0.5)
    parser.add_argument("--loop_weight", type=float, default=1)
    parser.add_argument("--extend_self_loop", default=False, type=str2bool)
    parser.add_argument("--internal_self_loop", default=False, type=str2bool)
    parser.add_argument("--w_max", type=float, default=1.0)
    parser.add_argument("--eps_bound", type=float, default=1e-3)
    parser.add_argument("--filter_mode", default="hard", choices=["gated", "sigmoid", "dual_sigmoid", "hard"])
    parser.add_argument("--prop_steps", type=int, default=2)
    parser.add_argument("--cls_aggr", default="sum")
    parser.add_argument("--tau_s", default=15, type=int)
    parser.add_argument("--tau_f", default=5, type=int)
    parser.add_argument("--cond_eval_step", default=2, type=float)

    parser.add_argument(
        "--dt_loss",
        default="dual",
        choices=["coarse_disc", "fine_disc", "dist_match", "contrast", "dual", "dual_contrast"],
    )
    parser.add_argument("--w_disc", default=1.0, type=float)
    parser.add_argument("--w_div", default=0.0, type=float)
    parser.add_argument("--n_pos_fine", default=1, type=int)
    parser.add_argument("--n_neg_fine", default=5, type=int)
    parser.add_argument("--neg_scale", default=2.0, type=float)
    parser.add_argument("--is_hard_neg", default=False)
    parser.add_argument("--n_cand_ratio", default=2, type=int)
    parser.add_argument("--temp", default=0.6, type=float)
    parser.add_argument("--es_top", default=5, type=int)

    parser.add_argument("--lr_x", default=1e-4, type=float)
    parser.add_argument("--lr_phi", default=1e-3, type=float)
    parser.add_argument("--lr_eps", default=1e-3, type=float)
    parser.add_argument("--dynamic", default="forward", choices=["forward", "reverse", "vanilla"])
    parser.add_argument("--w_c", default=1.0, type=float)
    parser.add_argument("--w_f", default=1.0, type=float)

    parser.add_argument("--add_self_loop_fast", default=True)

    parser.set_defaults(add_self_loop=False)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(HCHA_symdegnorm=False)

    args = parser.parse_args()
    return args
