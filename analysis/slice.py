import torch
from analysis.grouping import GroupData
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
import joblib
from scipy.sparse import csr_matrix

# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
device = ('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    idx = sorted(list(sub.SM))
    aud_slice = slice(0, 175)
    reduced = sub[:, :, :, idx]['zscore'][:, ('aud_ls', 'resp')]
    reduced.array = reduced.array.dropna()
    reduced = reduced.nan_common_denom(True, 10, True)
    # transfer data to torch tensor
    zscores = reduced['aud_ls'].array[:,:,:, aud_slice].concatenate(reduced['resp'].array, 3)
    data = zscores.combine((0, 2)).__array__()
    # del sub

    stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                          # sub.signif['aud_lm', :, aud_slice],
                          sub.signif['resp', :]])
    neural_data_tensor = torch.tensor(
        data / np.std(data, axis=0),
        device=device, dtype=torch.long)

    # Assuming neural_data_tensor is your 3D tensor
    # Remove NaN values
    # mask = torch.isnan(neural_data_tensor)
    # neural_data_tensor[mask.any(dim=2)] = 0.
    #
    # # Convert to sparse tensor
    # sparse_tensor = neural_data_tensor.to_sparse(sparse_dim=2)
    # del data

    ## set up the model

    train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
                                                fraction_test=0.2,
                                                device=device)

    procs = 2
    torch.set_num_threads(joblib.cpu_count() // procs)
    loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                min_ranks = [2, 0, 0],
                                                max_ranks = [5, 2, 2],
                                                sample_size=4,
                                                mask_train=train_mask,
                                                mask_test=test_mask,
                                                processes_grid=procs,
                                                seed=1,
                                                min_std=10**-4,
                                                learning_rate=5*10**-3,
                                                max_iter=10**4,
                                                positive=True)