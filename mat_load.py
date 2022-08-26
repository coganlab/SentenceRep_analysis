from scipy.io import loadmat


def load_all(filename: str):
    d = loadmat(filename, simplify_cells=True)
    t = d['Task']
    z = d['allSigZ']
    a = d['sigMatA']
    sCh = d['sigChans']
    sChL = d['sigMatChansLoc']
    sChN = d['sigMatChansName']
    return t, z, a, sCh, sChL, sChN


Task, sigZ, sigA, sigChans, sigMatChansLoc, sigMatChansName = load_all('../pydata.mat')