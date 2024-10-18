# %% Imports
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from analysis.data import LabeledData
import torch
import numpy as np
from analysis.decoding.models import CNNTransformer, SimpleDecoder
from analysis.grouping import GroupData
import os
from ieeg.calc.mat import Labels, LabeledArray
from ieeg.viz.ensemble import plot_dist
import pickle

device = ('cuda' if torch.cuda.is_available() else 'cpu')
# %% Define data module

def process_data(data, n_iters, n_folds, val_size,
                 target_map, max_epochs, learning_rate = 1e-4, verbose=False):

    dm = LabeledData(data, n_folds, val_size, target_map)

    iter_accs = []
    es_pat = max_epochs // 40
    num_classes = len(target_map)
    for i in range(n_iters):
        dm.setup()

        fold_accs = []
        for fold in range(n_folds):
            dm.set_fold(fold)
            # print(dm.current_fold)
            # d_model = data.shape[0] * data.shape[2]
            # instantiate the model
            # in_channels = dm.get_data_shape()[-1]
            model = SimpleDecoder(num_classes, data.shape[0] * data.shape[2],
            learning_rate)
            # model = CNNTransformer(in_channels, num_classes, d_model,
            #                        kernel_size, stride, padding, n_head,
            #                        num_layers, dim_fc, dropout, learning_rate)
            # model.current_fold = fold
            callbacks = [
                EarlyStopping(monitor='val_loss')]
            trainer = L.Trainer(max_epochs=max_epochs,
                                # gradient_clip_val=gclip_val,
                                accelerator=device,
                                callbacks=callbacks,
                                logger=False,
                                enable_model_summary=verbose,
                                enable_progress_bar=verbose,
                                enable_checkpointing=False
                                )
            trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                        val_dataloaders=dm.val_dataloader())
            # print(trainer.logged_metrics)
            trainer.test(model, dm.test_dataloader(), verbose=False)
            fold_accs.append(trainer.logged_metrics['test_acc'])

            # save loss information
            # loss_dict = trainer.logger.metrics
            # loss_dict['fold'] = fold
            # loss_dict['model'] = model
        # print(f'Averaged accuracy: {sum(fold_accs) / len(fold_accs)}')
        iter_accs.append(fold_accs)
    # print(sum(iter_accs) / len(iter_accs), iter_accs)
    # print(iter_accs)
    return torch.as_tensor(iter_accs)

if __name__ == '__main__':

    # dummy data
    n_samples = 144
    n_timepoints = 200
    n_features = 111
    fs = 100

    # create the data module
    batch_size = 32
    n_folds = 5
    val_size = 0.2
    fpath = os.path.expanduser("~/Box/CoganLab")
    # # Create a gridspec instance with 3 rows and 3 columns
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    idxs = {'Auditory': sub.AUD, 'Sensory-Motor': sub.SM, 'Production': sub.PROD}
    out = {}
    colors = ['r', 'g', 'b']

    for i, (name, idx) in enumerate(idxs.items()):
        idx = sorted(list(idx))
        aud_slice = slice(0, 175)
        reduced = sub[:, :, :, idx][:, ['aud_ls', 'aud_lm']]
        # reduced = sub[:, :, :, idx][:, ['go_ls', 'go_lm']]

        reduced = reduced.nan_common_denom(True, 10, True)
        aud = reduced.array['zscore']

        idx = [i for i, l in enumerate(sub.array.labels[3]) if
         l in reduced.array.labels[2]]
        # transfer data to torch tensor
        aud.labels[0] = Labels(aud.labels[0].replace("aud_", ""))
        aud = LabeledArray(np.ascontiguousarray(aud.__array__()), aud.labels)
        data = aud.combine((0,1)).combine((0,2)).dropna()
        # aud_go = LabeledArray(np.ascontiguousarray(aud_go.__array__()), aud_go.labels)
        # data = aud_go.combine((0, 1)).combine((0, 2)).__array__()
        # del sub

        stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                              # sub.signif['aud_lm', :, aud_slice],
                              sub.signif['resp', :]])

        neural_data_tensor = torch.from_numpy(
            data / np.nanstd(data)).to(device, dtype=torch.float32).permute(1, 2, 0)

        # %% Create model
        # model parameters
        in_channels = data.shape[0]
        # num_classes = 4
        d_model = 12
        kernel_time = 50  # ms
        kernel_size = int(kernel_time * fs / 1000)  # kernel length in samples
        stride_time = 10 # ms
        stride = int(stride_time * fs / 1000)  # stride length in samples
        padding = 0
        n_head = 4
        num_layers = 2
        dim_fc = 24
        dropout = 0.3
        # learning_rate = 1e-5

        # instantiate the model
        # model = CNNTransformer(in_channels, num_classes, data.shape[1], kernel_size, stride, padding,
        #                           n_head, num_layers, dim_fc, dropout, learning_rate)

        # %% Train the model with kfold
        # instantiate the trainer
        n_folds = 5
        max_epochs = 1000

        callbacks = [
                     EarlyStopping(monitor='val_loss', patience=3, mode='min',min_delta=0.0001
                                   )]
        target_map = {'heat': 0, 'hut': 1, 'hot': 2, 'hoot': 3}


        # %% train the model

        # out = process_data(data[..., :20], 1, n_folds, val_size, target_map, max_epochs, verbose=False)

        # %% windowed decoding
        from analysis.decoding import windower, Decoder
        from joblib import Parallel, delayed

        data_windowed = LabeledArray(windower(data, 20, 2).swapaxes(0, -1))[::4]
        data_windowed.labels[1] = data.labels[0]
        data_windowed.labels[2] = data.labels[1]

        out[name] = Parallel(n_jobs=4, verbose=40)(delayed(process_data)(
            d, 10, n_folds, val_size, target_map, max_epochs) for d in data_windowed)


    with open('decoding_results_aud.pkl', 'wb') as f:
        pickle.dump(out, f)
    # %% plot the results
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    for i, (name, idx) in enumerate(idxs.items()):
        plot = torch.stack(out[name]).flatten(1).T
        fig = plot_dist(plot.detach().numpy(), times=(-0.4, 1.4), mode='std', ax=ax, color=colors[i], label=name)
    fig.title.set_text("Decoding accuracy")
