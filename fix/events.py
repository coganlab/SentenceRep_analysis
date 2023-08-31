import mne
from ieeg import Signal
import pandas as pd
import numpy as np


def fix_annotations(inst: Signal):
    """fix SentenceRep events"""
    is_sent = False
    is_bad = False
    annot = None
    no_response = []
    for i, event in enumerate(inst.annotations):

        if 'boundary' in event['description']:
            event.pop('orig_time')
            annot.append(**event)
            continue
        # check if sentence or word trial
        if event['description'].strip() in ['Audio']:
            if event['duration'] > 1:
                is_sent = True
            else:
                is_sent = False

        # check if trials co-occur and mark bad
        if i != 0:
            prev = inst.annotations[i-1]
            if prev['onset'] + prev['duration'] > event['onset'] and \
                    prev['onset'] == annot[-1]['onset']:
                annot.description[-1] = 'bad ' + annot[-1]['description']
                mne.utils.logger.warn(f"Condition {i-1} and {i} co-occur")
                is_bad = True

        # check for trial type or bad
        if event['description'].strip() not in ['Listen', ':=:']:
            if is_bad or 'bad' in event['description'].lower():
                trial_type = "bad "
            elif is_sent:
                trial_type = "Sentence/"
            else:
                trial_type = "Word/"
        else:
            # determine trial type
            trial_type = "Start/"
            is_bad = False
            if event['description'].strip() in [':=:']:
                cond = "/JL"
            elif 'Mime' in inst.annotations[i + 2]['description']:
                cond = "/LM"
            elif event['description'].strip() in ['Listen']:
                cond = "/LS"
                if 'Speak' not in inst.annotations[i + 2]['description']:
                    if 'Response' in inst.annotations[i + 2]['description']:
                        mne.utils.logger.warn(f"Early response condition {i}")
                    else:
                        mne.utils.logger.error(
                            f"Speak cue not found for condition #{i} "
                            f"{event['description']}")
                    is_bad = True
                if len(inst.annotations) < i+4:
                    is_bad = True
                    no_response.append(i)
                elif 'Response' not in inst.annotations[i + 3]['description']:
                    is_bad = True
                    no_response.append(i)

            else:
                raise ValueError("Condition {} could not be determined {}"
                                 "".format(i, event['description']))
        if 'GoCue' in event['description']:
            event['description'] = 'Go'

        event['description'] = trial_type + event['description'] + cond
        if annot is None:
            annot = mne.Annotations(**event)
        else:
            event.pop('orig_time')
            annot.append(**event)
    inst.set_annotations(annot)
    return no_response


def add_stim_conds(inst: Signal):
    """Read the events files and add stim label to Audio events"""
    e_fnames = [f.replace('ieeg.edf', 'events.tsv') for f in inst.filenames]

    # read all the events files into a dataframe and concatenate
    df = pd.concat([pd.read_csv(f, sep='\t') for f in e_fnames],
                   ignore_index=True)

    # make sure the number of stim labels matches the number of audio events
    stim_files = df['stim_file'].str.contains('.wav', na=False).tolist()
    aud_events = ['Audio' in d for d in inst.annotations.description]
    if not sum(stim_files) == sum(aud_events):
        raise ValueError(f"Number of stim files ({stim_files}) does not match "
                         f"number of audio events ({aud_events})")

    # add stim labels to audio events
    stim_labels = df.loc[stim_files, 'stim_file'].tolist()
    stim_labels = ['/' + s.replace('.wav', '') for s in stim_labels]
    annot = None
    stim = None
    for event in inst.annotations:
        if 'Audio' in event['description']:
            stim = stim_labels.pop(0)
            event['description'] += stim
            annot[-1]['description'] += stim
        elif any(w in event['description'] for w in ['Word', 'Sentence']
                 ) and stim is not None:
            event['description'] += stim

        if annot is None:
            annot = mne.Annotations(**event)
        else:
            event.pop('orig_time')
            annot.append(**event)
    inst.set_annotations(annot)
    return inst
