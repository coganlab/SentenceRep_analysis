from scipy.io import loadmat
from numpy import where, unique


def load_all(filename: str):
    d = loadmat(filename, simplify_cells=True)
    t = d['Task']
    z = d['allSigZ']
    a = d['sigMatA']
    sCh = d['sigChans']
    sChL = d['sigMatChansLoc']
    sChN = d['sigMatChansName']
    sub = d['Subject']
    return t, z, a, sCh, sChL, sChN, sub


def group_elecs(sigA, sig_chans):
    # [AUDLS,] = where([1 in times[50: 174] for times in sigA['LSwords']['AuditorywDelay']])
    # AUDLS = [A for A in AUDLS if A in sigChans['LSwords']['AuditorywDelay']]
    AUD = dict()
    for cond in ['LS', 'LM', 'JL']:
        condw = cond + 'words'
        [ad,] = where([1 in times[50: 174] for times in sigA[condw]['AuditorywDelay']])
        AUD[cond] = [A for A in ad if A in sigChans[condw]['AuditorywDelay']]
        # AUD[cond] = where(sigA[condw]['AuditorywDelay'][:, 50: 174] == 1)
        # AUD[cond] = unique(AUD[cond])
        # AUD[cond] = set(AUD[cond]) & set(sig_chans[condw]['AuditorywDelay'])

    AUD1 = set(AUD['LS']) & set(AUD['LM'])
    AUD2 = set(AUD1) & set(AUD['JL'])
    PROD1 = set(sig_chans['LSwords']['DelaywGo']) & set(sig_chans['LMwords']['DelaywGo'])
    SM = list(set(AUD1) & set(PROD1))
    AUD = list(set(AUD2) - set(SM))
    PROD = list(set(PROD1) - set(SM))
    return SM, AUD, PROD


if __name__ == "__main__":
    Task, sigZ, sigA, sigChans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
    SM, AUD, PROD = group_elecs(sigA, sigChans)