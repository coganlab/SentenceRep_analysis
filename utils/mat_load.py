from scipy.io import loadmat
from numpy import where, concatenate
from utils.calc import ArrayLike


def load_all(filename: str):
    d = loadmat(filename, simplify_cells=True)
    t = d['Task']
    z = d['allSigZ']
    a = d['sigMatA']
    # reduce indexing by 1 because python is 0-indexed
    for cond, epochs in d['sigChans'].items():
        for epoch, vals in epochs.items():
            d['sigChans'][cond][epoch] = vals - 1
    sCh = d['sigChans']
    sChL = d['sigMatChansLoc']
    sChN = d['sigMatChansName']
    sub = d['Subject']
    return t, z, a, sCh, sChL, sChN, sub


def group_elecs(sigA: dict[str, dict[str, ArrayLike]], sig_chans: dict[str, dict[str, list[int]]]
                ) -> tuple[list, list, list]:
    AUD = dict()
    for cond in ['LS', 'LM', 'JL']:
        condw = cond + 'words'
        elecs = where(sigA[condw]['AuditorywDelay'][:, 50:175] == 1)[0]
        AUD[cond] = set(elecs) & set(sig_chans[condw]['AuditorywDelay'])

    AUD1 = AUD['LS'] & AUD['LM']
    AUD2 = AUD1 & AUD['JL']
    PROD1 = set(sig_chans['LSwords']['DelaywGo']) & set(sig_chans['LMwords']['DelaywGo'])
    SM = list(AUD1 & PROD1)
    AUD = list(AUD2 - set(SM))
    PROD = list(PROD1 - set(SM))
    for group in [SM, AUD, PROD]:
        group.sort()
    return SM, AUD, PROD


def get_sigs(allsigZ: dict[str, dict[str, ArrayLike]], allsigA: dict[str, dict[str, ArrayLike]],
             sigChans: dict[str, dict[str, list[int]]], cond: str) -> tuple[dict[str, ArrayLike], dict[str, ArrayLike]]:
    out_sig = dict()
    for sig, metric in zip([allsigZ, allsigA], ['Z', 'A']):
        out_sig[metric] = dict()
        for group, idx in zip(['SM', 'AUD', 'PROD'], group_elecs(allsigA, sigChans)):
            blend = sig[cond]['AuditorywDelay'][idx, 150:175] / 2 + \
                    sig[cond]['DelaywGo'][idx, 0:25] / 2
            out_sig[metric][group] = concatenate((sig[cond]['AuditorywDelay'][idx, :150],
                                                  blend,
                                                  sig[cond]['DelaywGo'][idx, 25:]), axis=1)

    return out_sig['Z'], out_sig['A']


if __name__ == "__main__":
    Task, sigZ, sigA, sigChans, sigMatChansLoc, sigMatChansName, Subject = load_all('../data/pydata.mat')
    SM, AUD, PROD = group_elecs(sigA, sigChans)
