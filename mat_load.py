from scipy.io import loadmat
from numpy import where, concatenate


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


def group_elecs(sigA, sig_chans):
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
    return SM, AUD, PROD


def get_sigs(allsigZ, allsigA, sigChans):
    sigZ = dict()
    sigA = dict()
    for group, idx in zip(['SM', 'AUD', 'PROD'], group_elecs(allsigA, sigChans)):
        sigZ[group] = concatenate((allsigZ['LSwords']['AuditorywDelay'][idx, :175],
                                  allsigZ['LSwords']['DelaywGo'][idx]), axis=1)
        sigA[group] = concatenate((allsigA['LSwords']['AuditorywDelay'][idx, :175],
                                  allsigA['LSwords']['DelaywGo'][idx]), axis=1)
    return sigZ, sigA


if __name__ == "__main__":
    Task, sigZ, sigA, sigChans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
    SM, AUD, PROD = group_elecs(sigA, sigChans)