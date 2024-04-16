import mne
from ieeg import Signal
import pandas as pd
import numpy as np


def fix_annotations(inst: Signal):
    """fix SentenceRep events"""
    is_sent = False
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

        # check for trial type or bad
        if event['description'].strip() not in ['Listen', ':=:']:
            if is_sent:
                trial_type = "Sentence/"
            else:
                trial_type = "Word/"
        else:
            # determine trial type
            trial_type = "Start/"
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
                if len(inst.annotations) < i+4:
                    no_response.append(i)
                elif 'Response' not in inst.annotations[i + 3]['description']:
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


def get_overlapping_indices(onsets, durations, margin=0):
    # Calculate the end times for each event
    end_times = onsets + durations

    # Initialize an empty list to store the indices of overlapping events
    overlapping_indices = set()

    # Iterate over each event
    for i in range(len(onsets)):
        # Check if the start time of the current event is within the duration of any other event
        for j in range(len(onsets)):
            if i != j and onsets[i] >= onsets[j] - margin and onsets[i] <= end_times[j] + margin:
                overlapping_indices.add(i)
                overlapping_indices.add(j)
                break

    return sorted(list(overlapping_indices))


def mark_bad(inst: Signal, bads: list[int, ...]):
    """Mark bad events"""
    annot = None
    is_bad = False
    for i, event in enumerate(inst.annotations):
        desc = event['description']
        if 'boundary' in desc:
            event.pop('orig_time')
            annot.append(**event)
            continue
        elif desc.startswith('bad'):
            is_bad = True
        elif desc.startswith('Start') and i not in bads:
            is_bad = False
        elif i in bads or is_bad:
            event['description'] = 'bad ' + desc
            is_bad = True

        if annot is None:
            annot = mne.Annotations(**event)
        else:
            event.pop('orig_time')
            annot.append(**event)
    inst.set_annotations(annot)
    return inst


def fix(inst: Signal):
    """Fix the events"""
    no_response = fix_annotations(inst)
    bad = get_overlapping_indices(inst.annotations.onset,
                                  inst.annotations.duration, 0.15)
    bad += no_response
    inst = add_stim_conds(inst)
    inst = mark_bad(inst, bad)
    return inst


if __name__ == "__main__":
    import os
    from ieeg.io import get_data, raw_from_layout

    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")

    for subj in subjects:
        # if int(subj[1:]) != 28:
        #     continue
        raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                                preload=True)
        filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                                extension='.edf', desc='clean', preload=False)
        fixed = fix(raw)
        fixed.annotations._orig_time = filt.annotations.orig_time
        filt.set_annotations(fixed.annotations)


        files = layout.derivatives['clean'].get(subject=subj, suffix='events',
                                extension='.tsv', desc='clean')

        i = 0
        data = []
        offset = 0
        for f in files:
            events = f.get_df()
            for j, event in events.iterrows():
                if 'boundary' in event['trial_type']:
                    if event['trial_type'] == 'BAD boundary':
                        continue
                    offset = filt.annotations.onset[i] - event['onset']
                    i += 1
                    continue
                diff = abs(event['onset'] - filt.annotations.onset[i] + offset)
                if diff > 0.15:
                    raise ValueError(f"{filt.annotations[i]} is not aligned with {event}, diff={diff}")
                events.loc[j, 'trial_type'] = filt.annotations.description[i]
                i += 1
            data.append(events)
        for f, events in zip(files, data):
            events.to_csv(f.path, sep='\t', index=False)
