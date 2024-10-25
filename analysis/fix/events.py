import mne
from ieeg import Signal
import pandas as pd
import dataclasses
import itertools
from ieeg.viz.ensemble import chan_grid
from ieeg.timefreq.multitaper import spectrogram
from ieeg.calc.scaling import rescale
from ieeg.timefreq.utils import crop_pad
from ieeg.navigate import channel_outlier_marker, trial_ieeg, outliers_to_nan, crop_empty_data
from ieeg.viz.parula import parula_map
import numpy as np

@dataclasses.dataclass(frozen=True, order=True, slots=True)
class Event:
    """A class to represent an Event."""
    onset: float
    duration: float
    description: str = dataclasses.field(compare=False)
    bad: bool = dataclasses.field(default=False, compare=False)
    why: str = dataclasses.field(default="", compare=False)

    def __post_init__(self):
        """Validates the data after initialization."""
        self.validate()

    def __str__(self):
        return (f"{self.description}: {self.onset:.4f} - "
                f"{self.onset + self.duration:.4f} s. {self.why}").strip()

    def __repr__(self):
        return self.__str__()

    def validate(self):
        """Validates the onset, duration, and description."""
        if self.onset < 0:
            raise ValueError("onset must be positive")
        if self.duration < 0:
            raise ValueError("duration must be positive")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        if not self.description:
            raise ValueError("description must not be empty")

    def mark_event_as_bad(self, why: str) -> 'Event':
        """Marks the event as bad."""
        if self.bad:
            return self
        return dataclasses.replace(self, bad=True, why=why,
                                   description='bad ' + self.description)

    def relabel(self, new_desc: str) -> 'Event':
        """Relabels the event."""
        out = dataclasses.replace(self, description=new_desc, bad=False)
        if self.bad:
            return out.mark_event_as_bad(self.why)
        return out


@dataclasses.dataclass(order=True)
class Trial:
    """A class to represent a Trial."""
    start: Event
    stim: Event
    go: Event
    response: Event = None

    def __post_init__(self):
        """Validates the data after initialization."""
        self.validate_events()
        self.check_co_occurrence()
        self.relabel_events()

    def validate_events(self):
        """Validates the identity of each event based on the description."""
        events = [('start', ['Listen', ':=:'], "Start event"),
                  ('stim', ['Audio'], "Stim event"),
                  ('go', ['Speak'], "Go event"),
                  ('response', ['Response'], "Response event")]

        for event_name, valid_descriptions, error_message in events:
            event = getattr(self, event_name)
            if event is not None:
                description = event.description.replace("bad ", "").split("/")
                if not any(i in description for i in valid_descriptions):
                    raise ValueError(f"{error_message} {event} is not a "
                                     f"valid {event_name} event")

    def check_co_occurrence(self):
        """Checks the co-occurrence of events."""
        pairs = (('start', 'stim'), ('stim', 'go'), ('go', 'response'))
        for first, second in pairs:
            prev = getattr(self, first)
            current = getattr(self, second)

            if second == 'response' and current is None:
                # mark all no response go cues as bad
                if self.condition == 'LS':
                    setattr(self, first, prev.mark_event_as_bad("No response"))
                continue
            elif prev.onset + prev.duration > current.onset:
                setattr(self, first, prev.mark_event_as_bad(f"Overlapping {second}"))
                setattr(self, second, current.mark_event_as_bad(f"Overlapping {first}"))

    def relabel_events(self):
        """Relabels the events."""
        new = "/".join(('Start', self.start.description, self.condition))
        setattr(self, 'start', self.start.relabel(new.replace('/bad ', '/')))

        new = "/".join(
            (self.trial_type, self.stim.description, self.condition))
        setattr(self, 'stim', self.stim.relabel(new.replace('/bad ', '/')))

        new = "/".join((self.trial_type, 'Go', self.condition))
        setattr(self, 'go', self.go.relabel(new))

        if self.response is not None:
            new = "/".join((self.trial_type, 'Response', self.condition))
            setattr(self, 'response', self.response.relabel(new))
            if self.go.bad:
                setattr(self, 'response', self.response.mark_event_as_bad("bad go"))

    @property
    def trial_type(self) -> str:
        """Returns the trial type."""
        if self.stim.duration > 1.:
            return 'Sentence'
        else:
            return 'Word'

    @property
    def condition(self) -> str:
        """Returns the condition."""
        all_desc = self.start.description.split('/')
        for cond in ("LS", "LM", "JL"):
            if cond in all_desc:
                return cond

        if ':=:' in self.start.description:
            return "JL"
        elif 'Listen' in self.start.description:
            if 'Mime' in self.go.description:
                return "LM"
            elif 'Speak' in self.go.description: #when applied to phoneme seq, this should make all Trial.condition = "LS"
                return "LS"
            else:
                raise ValueError("Condition could not be determined")
        else:
            raise ValueError("Condition could not be determined")

    def mark_bad(self, event: str = None, why: str = "bad trial"):
        """Marks the whole trial as bad."""
        if event is None:
            self.start = self.start.mark_event_as_bad(why)
            self.stim = self.stim.mark_event_as_bad(why)
            self.go = self.go.mark_event_as_bad(why)
            if self.response is not None:
                self.response = self.response.mark_event_as_bad(why)
        else:
            setattr(self, event, getattr(self, event).mark_event_as_bad(why))

    def get_events(self) -> tuple[Event, ...]:
        """Returns the events in the trial."""
        if self.response is None:
            return self.start, self.stim, self.go
        else:
            return self.start, self.stim, self.go, self.response

    @classmethod
    def chunk(cls, iterable: list[Event]) -> tuple['Trial', ...]:

        starts, stims, gos, responses = [], [], [], []
        for event in iterable:
            if 'Listen' in event.description or ':=:' in event.description:
                starts.append(event)
            elif 'Audio' in event.description:
                stims.append(event)
            elif 'Speak' in event.description:
                gos.append(event)
            elif 'Response' in event.description:
                responses.append(event)
            else:
                raise ValueError(f"Unknown event {event}")

        assert len(starts) == len(stims) == len(gos), "Unequal number of events"

        while starts:
            start = starts.pop(0)
            stim = stims.pop(0)
            go = gos.pop(0)
            if not responses:
                yield cls(start, stim, go)
            elif responses[0].onset < stims[0].onset if stims else float("inf"):
                yield cls(start, stim, go, responses.pop(0))
            else:
                yield cls(start, stim, go)


def fix_annotations(annotations: mne.Annotations, events: list[Event, ...]):
    """fix SentenceRep events"""
    annot = None
    for a in annotations:

        orig = a.pop('orig_time')
        if 'boundary' in a['description']:
            annot.append(**a)
            continue
        else:
            event = events.pop(0)
            assert Event(**a) == event, f"onset mismatch {a['onset']} != {event.onset}"
            # assert a['duration'] - event.duration < 0.001, f"duration mismatch {a['duration']} != {event.duration}"
            a['description'] = event.description

        if annot is None:
            annot = mne.Annotations(**a, orig_time=orig)
        else:
            annot.append(**a)
    if len(events) > 0:
        raise ValueError(f"More events than annotations {len(events)}")
    return annot


def add_stim_conds(inst: Signal):
    """Read the events files and add stim label to Audio events"""
    e_fnames = (f.replace('ieeg.edf', 'events.tsv') for f in inst.filenames)

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


def fix(inst: Signal):
    """Fix the events"""
    items = zip(inst.annotations.onset, inst.annotations.duration,
                inst.annotations.description)
    events = [Event(*i, bad=i[2].startswith('bad')) for i in items if 'boundary' not in i[2]]
    trials = [trial for trial in Trial.chunk(events)]

    # mark bad trials using custom logic
    for i, trial in enumerate(trials):
        if trial.response is None and trial.condition == 'LS':
            trial.mark_bad(why='No response')
        elif trial.response is not None and trial.condition != 'LS':
            trial.mark_bad(why='Response in non-LS trial')

        # if i == 0:
        #     continue
        # elif trials[i - 1].trial_type == 'Sentence':
        #     trial.mark_bad('start', 'Prior Sentence')
        #     trial.mark_bad('stim', 'Prior Sentence')

    events_sorted = sorted(
        itertools.chain.from_iterable((t.get_events() for t in trials)))

    if len(events_sorted) != len(events):
        diff = set(events) - set(events_sorted)
        for i in diff:
            events_sorted.append(i.mark_event_as_bad("Unpaired"))
        events_sorted.sort()

    annotations = fix_annotations(inst.annotations, events_sorted.copy())
    inst.set_annotations(annotations)
    inst = add_stim_conds(inst)
    return inst


if __name__ == "__main__":
    import os
    from ieeg.io import get_data, raw_from_layout

    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("Phoneme_sequencing", root=LAB_root)
    subjects = layout.get(return_type="id", target="subject")

    for subj in subjects:
        if subj != subjects[15]:
            continue
        raw = raw_from_layout(layout, subject=subj, extension=".edf",
                              desc=None, preload=True)
        filt = raw_from_layout(layout.derivatives['clean'], subject=subj,
                               extension='.edf', desc='clean', preload=True)

        #fixed = fix(raw.copy())
        # fixed.annotations._orig_time = fixed.annotations.orig_time
        # fixed.set_annotations(fixed.annotations)
        _, ids = mne.events_from_annotations(raw, regexp='.*')

        ## Crop raw data to minimize processing time
        good = crop_empty_data(raw, ).copy()

        good.info['bads'] = channel_outlier_marker(good, 3, 2)
        good.drop_channels(good.info['bads'])
        good.load_data()


        # CAR
        # ch_type = good.get_channel_types(only_data_chs=True)[0]
        # good.set_eeg_reference(ref_channels="average", ch_type=ch_type)
        #
        # freq = np.arange(10, 200., 6.)
        # base = trial_ieeg(good, 'Listen', (-1, 0.5), preload=True)
        # outliers_to_nan(base, outliers=10)
        # trials = trial_ieeg(good, 'Response', (-1.2, 1.2), preload=True)
        # outliers_to_nan(trials, outliers=10)
        #
        # kwargs = dict(average=False, n_jobs=20, freqs=freq, return_itc=False,
        #               n_cycles=freq / 2, time_bandwidth=10,
        #               # n_fft=int(trials.info['sfreq'] * 2.75),
        #               decim=20, )
        #
        # spectra = trials.compute_tfr(method="multitaper", **kwargs)
        # base_spectra = base.compute_tfr(method="multitaper", **kwargs)
        # del trials
        # base = base_spectra.crop(-0.5, 0).average(lambda x: np.nanmean(x, axis=0), copy=True)
        #
        # spectra = spectra.average(lambda x: np.nanmean(x, axis=0), copy=True)
        # rescale(spectra._data, base._data, mode='ratio', axis=-1)
        # crop_pad(spectra, "0.5s")
        # chan_grid(spectra, vlim=(0.7, 1.4), cmap=parula_map)


        # wrapper that doesn't work
        # spectra = spectrogram(raw, freq, 'Response', -1.2, 1.2, 'Listen', -0.5, 0.,
        #                       n_jobs=-2, verbose=40, time_bandwidth=10, n_cycles=freq / 2)

        ## OLD CODE to check potential annotation issues, now no longer used as bad boundary checked in class method
        # files = layout.derivatives['clean'].get(subject=subj, suffix='events',
        #                         extension='.tsv', desc='clean')
        #     raise ValueError("This is a test")
        #     i = 0
        #     data = []
        #     offset = 0
        #     for f in files: # this gets all runs for the subject
        #     events = f.get_df() # this gets df of onset, duration, trial_type, value?, and sample
        #         for j, event in events.iterrows():
        #             if 'boundary' in event['trial_type']:
        #                 if event['trial_type'] == 'BAD boundary':
        #                     continue
        #                 offset = filt.annotations.onset[i] - event['onset']
        #                 i += 1
        #                 continue
        #             diff = abs(event['onset'] - filt.annotations.onset[i] + offset)
        #             if diff > 0.15:
        #                 raise ValueError(
        #                     f"{filt.annotations[i]} is not aligned with {event}, diff={diff}")
        #             events.loc[j, 'trial_type'] = filt.annotations.description[i]
        #             events.loc[j, 'value'] = ids[filt.annotations.description[i]]
        #             i += 1
        #         data.append(events)
        #     for f, events in zip(files, data):
        #         events.to_csv(f.path, sep='\t', index=False)
        # except Exception as e:
        #     print(f"Skipping {subj} due to {e}")
        #     raise e
