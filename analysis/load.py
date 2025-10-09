import os
import torch
import numpy as np
from ieeg.arrays.label import LabeledArray, combine
from analysis.grouping import group_elecs, GroupData
from analysis.data import dataloader
from analysis.decoding.utils import extract
from ieeg.io import DataLoader
from ieeg import Doubles

# Exclude list for bad channels
exclude = [
    "D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
    "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
    "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
    "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
    "D0027-LPIF10", "D0026-RPG20", "D0026-RPG21", "D0026-RPG28",
    "D0026-RPG29", "D0026-RPG36","D0026-RPST2", "D0007-RFG44"
]

def load_tensor(array, idx, conds, trial_ax, min_nan=1):
    idx = sorted(idx)
    X = extract(array, conds, trial_ax, idx, min_nan)
    # std = float(np.nanstd(X.__array__(), dtype='f8'))
    print("Calculating std...")
    std_ch = np.nanstd(X.__array__(), (0,2,3,4), dtype='f8')
    std = float(np.mean(std_ch))
    combined = np.concatenate([X[c] for c in conds], axis=-1)
    if not (goods := std_ch < (2 * std)).all():
        combined = combined[goods,]
        std = float(np.mean(std_ch[goods]))
    out_tensor = torch.from_numpy(combined.__array__() / std)
    mask = torch.isnan(out_tensor)
    return out_tensor, ~mask, list(map(list, combined.labels))

def load_spec(group, conds, layout, folder='stats_freq_hilbert',
              min_nan: int = 1, n_jobs: int = 1):
    sigs = load_data(layout, folder, "mask", conds, 2,
                     bool, True, n_jobs)
    AUD, SM, PROD, sig_chans, delay = group_elecs(sigs,
                                                  [s for s in
                                                   sigs.labels[1]
                                                   if s not in exclude],
                                                  sigs.labels[0])
    idxs = {'SM': sorted(SM), 'AUD': sorted(AUD), 'PROD': sorted(PROD),
            'sig_chans': sorted(sig_chans), 'delay': sorted(delay)}
    idx = idxs[group]
    zscores = load_data(layout, folder, "zscore", conds, 3,
                        "float16", False, n_jobs)
    neural_data_tensor, mask, labels = load_tensor(zscores, idx, conds, 4, min_nan)
    return neural_data_tensor, mask, labels, idxs

def load_hg(group, conds, LAB_root, **kwargs):
    sub = GroupData.from_intermediates("SentenceRep", LAB_root, folder='stats')
    idxs = {'SM': sorted(sub.SM), 'AUD': sorted(sub.AUD), 'PROD': sorted(sub.PROD),
            'sig_chans': sorted(sub.sig_chans), 'delay': sorted(sub.delay)}
    idx = idxs[group]
    neural_data_tensor, labels = dataloader(sub.array, idx, conds, **kwargs)
    neural_data_tensor = neural_data_tensor.swapaxes(0, 1).to(torch.float32)
    labels[0], labels[1] = labels[1], labels[0]
    mask = ~torch.isnan(neural_data_tensor)
    return neural_data_tensor, mask, labels, idxs

def split_and_stack(tensor, split_dim, stack_pos, num_splits, new_dim: bool = True):
    splits = torch.split(tensor, tensor.shape[split_dim] // num_splits, dim=split_dim)
    stacked = torch.stack(splits, dim=0)
    permute_order = list(range(stacked.ndim))
    permute_order.insert(stack_pos, permute_order.pop(0))
    out = stacked.permute(permute_order)
    if not new_dim:
        out = out.reshape(*out.shape[:stack_pos], -1, *out.shape[stack_pos + 2:])
    return out

def load_data(layout, folder: str, datatype: str,
              conds: dict[str, Doubles] = None, ch_dim: int = None,
              out_type: type | str = float, average: bool = True,
              n_jobs: int = 12, combined_folder: str = 'combined') -> LabeledArray:
    """
    Loads and combines zscore or other data for all subjects and conditions.
    Returns a LabeledArray.
    """
    filemask = os.path.join(layout.root, 'derivatives', folder,
                            combined_folder, datatype)
    if not os.path.exists(filemask + ".npy"):
        missing = [name for name, val in [('conds', conds),
                                          ('ch_dim', ch_dim)] if val is None]
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")

        loader = DataLoader(layout, conds, datatype, average, folder, '.h5')
        zscore = loader.load_dict(dtype=out_type, n_jobs=n_jobs)
        # Combine along subject and channel axis (0, 3)
        zscore_ave = combine(zscore, (0, ch_dim))
        zscores = LabeledArray.from_dict(zscore_ave, dtype=out_type)
        zscores.tofile(filemask)

    zscores = LabeledArray.fromfile(filemask, mmap_mode='r')
    return zscores
