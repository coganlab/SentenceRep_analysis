import os
from ieeg.io import get_data
from ieeg.arrays.label import LabeledArray
from analysis.grouping import GroupData
from analysis.data import dataloader
from analysis.decoding.utils import extract
from analysis.grouping import group_elecs
from itertools import product
import torch
import numpy as np
from functools import reduce
import slicetca
from multiprocessing import freeze_support
from functools import partial
from slicetca.run.dtw import SoftDTW as sdtw

class SoftDTW(sdtw):
    __module__ = "tslearn.metrics"
    def __repr__(self):
        return f"SoftDTW(gamma={self.gamma})"
    def __str__(self):
        return str(self.__repr__())

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
torch.set_float32_matmul_precision("medium")

def load_tensor(array, idx, conds, trial_ax, min_nan=5):
    idx = sorted(idx)
    X = extract(array, conds, trial_ax, idx, min_nan)
    std = float(np.nanstd(X.__array__(), dtype='f8'))
    std_ch = np.nanstd(X.__array__(), (0,2,3,4), dtype='f8')
    # mean = float(np.nanmean(X.__array__(), dtype='f8'))
    combined = reduce(lambda x, y: x.concatenate(y, -1),
                      [X[c] for c in conds])
    if (std_ch < (2 * std)).any():
        combined = combined[std_ch < (2 * std),]
    std = float(np.nanstd(combined.__array__(), dtype='f8'))
    out_tensor = torch.from_numpy(combined.__array__() / std)
    mask = torch.isnan(out_tensor)
    # n_nan = mask.sum(dtype=torch.int64)
    # out_tensor[mask] = torch.normal(mean, std, (n_nan,)).to(
    #     out_tensor.dtype)
    return out_tensor, ~mask, combined.labels

HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    n = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print(n)
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    n = -1

log_dir = os.path.join(os.path.dirname(LAB_root), 'logs', str(n))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
layout = get_data('SentenceRep', root=LAB_root)

conds_all = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                 "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                 "go_jl": (-0.5, 1.5)}

def load_spec(group, conds, folder='stats_freq_hilbert'):
    filemask = os.path.join(layout.root, 'derivatives', folder, 'combined',
                            'mask')
    sigs = LabeledArray.fromfile(filemask)
    AUD, SM, PROD, sig_chans, delay = group_elecs(sigs, sigs.labels[1],
                                                  sigs.labels[0])
    idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans,
            'delay': delay}
    idx = sorted(idxs[group])
    filename = os.path.join(layout.root, 'derivatives', folder, 'combined',
                            'zscore')
    zscores = LabeledArray.fromfile(filename, mmap_mode='r')
    neural_data_tensor, mask, labels = load_tensor(zscores, idx,
                                                   conds, 4, 1)
    return neural_data_tensor, mask, labels, idxs

def load_hg(group, conds, **kwargs):
    sub = GroupData.from_intermediates("SentenceRep", LAB_root, folder='stats')
    idxs = {'SM': sub.SM, 'AUD': sub.AUD, 'PROD': sub.PROD, 'sig_chans': sub.sig_chans,
            'delay': sub.delay}
    idx = sorted(idxs[group])
    neural_data_tensor, labels = dataloader(sub.array, idx, conds, **kwargs)
    neural_data_tensor = neural_data_tensor.swapaxes(0, 1).to(torch.float32)
    labels[0], labels[1] = labels[1], labels[0]
    mask = ~torch.isnan(neural_data_tensor)

    return neural_data_tensor, mask, labels, idxs

def split_and_stack(tensor, split_dim, stack_pos, num_splits, new_dim: bool = True):
    # Split tensor along split_dim
    splits = torch.split(tensor, tensor.shape[split_dim] // num_splits, dim=split_dim)
    # Stack splits into a new axis
    stacked = torch.stack(splits, dim=0)
    # Move new axis to stack_pos
    permute_order = list(range(stacked.ndim))
    permute_order.insert(stack_pos, permute_order.pop(0))
    out = stacked.permute(permute_order)
    if not new_dim:
        # new_dim is false, combine the new axis with the next axis
        out = out.reshape(*out.shape[:stack_pos], -1, *out.shape[stack_pos + 2:])
    return out

# %% grid search
pick_k = True
if pick_k:
    if __name__ == '__main__':
        freeze_support()
    param_grid = {'lr': [1e-2, 1e-3, 1e-4],
                'ranks': [{'min': [1, 0, 0], 'max': [9, 0, 0]},
                            {'min': [1], 'max': [9]},],
                  'groups': ['AUD', 'SM', 'PROD', 'sig_chans'],
                  'loss': ['L1Loss',
                           SoftDTW(True, 50, True, 20,
                                   torch.nn.L1Loss(reduction='none')),
                           SoftDTW(True, 1, True, 20,
                                   torch.nn.L1Loss(reduction='none')),
                           SoftDTW(True, .1, True, 20,
                                   torch.nn.L1Loss(reduction='none')),
                           'HuberLoss',
                           SoftDTW(True, 50, True, 20,
                                   torch.nn.HuberLoss(reduction='none')),
                           SoftDTW(True, 1, True, 20,
                                   torch.nn.HuberLoss(reduction='none')),
                           SoftDTW(True, .1, True, 20,
                                   torch.nn.HuberLoss(reduction='none')),
                           ],
                  'decay': [1],
                  'batch': [True, False],
                  'spec': [0, 1, 2]}
    procs = 1
    threads = 1
    repeats = 2
    conds = ['aud_ls', 'go_ls', 'aud_lm', 'go_lm', 'aud_jl', 'go_jl']
    aud_slice = slice(0, 175)

    for lr, ranks, group, loss, decay, batched, spec in product(
            param_grid['lr'], param_grid['ranks'], param_grid['groups'],
            param_grid['loss'], param_grid['decay'],
    param_grid['batch'], param_grid['spec']):
        if n > 1:
            n -= 1
            continue
        elif 0 <= n < 1:
            break
        else:
            n -= 1
            print(ranks, group, loss, lr, decay, batched, spec)

        rank_min = ranks['min']
        rank_max = ranks['max']
        if spec == 1:
            neural_data_tensor, mask, labels, idxs = load_spec(group, conds)
            trial_ax = 2
            train_blocks_dimensions = (1, 10, 10)  # Note that the blocks will be of size 2*train_blocks_dimensions + 1
            test_blocks_dimensions = (1, 5, 5)  # Same, 2*test_blocks_dimensions + 1
            if len(ranks['min']) > 1:
                rank_min = ranks['min'] + [0]
                rank_max = ranks['max'] + [0]
        elif spec == 0:
            neural_data_tensor, mask, labels, idxs = load_hg(group, conds)
            trial_ax = 1
            train_blocks_dimensions = (1, 10)
            test_blocks_dimensions = (1, 5)
        else:
            neural_data_tensor, mask, labels, idxs = load_spec(group, conds, 'stats_freq_multitaper')
            trial_ax = 2
            train_blocks_dimensions = (1, 10, 10)  # Note that the blocks will be of size 2*train_blocks_dimensions + 1
            test_blocks_dimensions = (1, 5, 5)  # Same, 2*test_blocks_dimensions + 1
            if len(ranks['min']) > 1:
                rank_min = ranks['min'] + [0]
                rank_max = ranks['max'] + [0]

        idx = sorted(idxs[group])

        kwargs = {'regularization': 'L2' if decay < 1 else None}
        if batched:
            kwargs['batch_dim'] = trial_ax + 1
            kwargs['shuffle_dim'] = (0, 1)
            kwargs['precision'] = '16-mixed'
            neural_data_tensor = neural_data_tensor.to(torch.float16)
        else:
            neural_data_tensor = neural_data_tensor.nanmean(trial_ax, dtype=torch.float32)

        ## set up the model
        if not batched:
            train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                        train_blocks_dimensions=train_blocks_dimensions, # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                        test_blocks_dimensions=test_blocks_dimensions, # Same, 2*test_blocks_dimensions + 1
                                                        fraction_test=0.2)
            # test_mask = torch.logical_and(test_mask, mask)
            # train_mask = torch.logical_and(train_mask, mask)
        else:
            train_mask = mask
            test_mask = None

        if isinstance(loss, str):
            loss_fn = getattr(torch.nn, loss)(reduction='mean')
        else:
            loss_fn = loss.__str__()

        file_id = (f"results_{group}_{'batched' if batched else 'unbatched'}_"
                   f"{'spec' if spec else 'HG'}_{len(rank_min)}ranks_{loss}_"
                   f"{lr}_{decay}.pkl")

        loss_grid, seed_grid = slicetca.grid_search(
            split_and_stack(neural_data_tensor, -1, 1, 3),
            min_ranks = rank_min,
            max_ranks = rank_max,
            sample_size=repeats,
            mask_train=split_and_stack(train_mask, -1, 1, 3) if train_mask is not None else None,
            mask_test=split_and_stack(test_mask, -1, 1, 3) if test_mask is not None else None,
            processes_grid=procs,
            processes_sample=threads,
            seed=3,
            batch_prop=decay,
            batch_prop_decay=3 if decay < 1 else 1,
            # min_std=1e-4,
            # iter_std=10,
            init_bias=0.01,
            weight_decay=partial(
                torch.optim.Adam,
                # betas=(0.5, 0.5),
                # amsgrad=True,
                eps=1e-10,
                # weight_decay=0
                ),
            initialization='uniform-positive',
            learning_rate=lr,
            max_iter=1000000,
            positive=True,
            verbose=0,
            loss_function=loss_fn,
            compile=True,
            min_iter=1,
            gradient_clip_val=1,
            default_root_dir=log_dir,
            dtype=torch.float32,
            fast_dev_run=True,
            checkpoint=file_id,
            **kwargs
        )

