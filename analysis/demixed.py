import os.path
from grouping import GroupData

fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats_opt')

conds_aud = ['aud_ls', 'aud_lm', 'aud_jl']
conds_go = ['go_ls', 'go_lm', 'go_jl']
aud = sub.array['zscore', conds_aud]
go = sub.array['zscore', conds_go]
aud_go = aud[..., :175].concatenate(go, -1)
