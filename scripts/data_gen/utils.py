import mne
from pathlib import Path
import pandas as pd
import math
from collections import defaultdict
mne.set_log_level('WARNING')

def populate_bdf_index(DATA_ROOT, filt_ids):
    bdf_index = defaultdict(list)
    for path in Path(DATA_ROOT).rglob("*.bdf"):
        subj_id = path.name.split("_")[0]
        if subj_id in filt_ids and "ses-1" in path.parts:
            bdf_index[subj_id].append(path)
    return bdf_index

def generate_string_neural_data(bdf_index):
    neural_data = {}
    for subj_index, (subj_id, bdf_paths) in enumerate(bdf_index.items()):
        record = ""
        for bdf_path in bdf_paths:
            band_powers, channel_names = process_bdf(bdf_path)
            if "restEC" in bdf_path.name:
                record += "Eyes Closed Spectral Features (μV²/Hz):\n"
            elif "restEO" in bdf_path.name:
                record += "Eyes Open Spectral Features (μV²/Hz):\n"
            else:
                print(subj_id)
                print(bdf_paths)
                continue
            for index, channel in enumerate(channel_names):
                    record += "Channel " + channel + ": "
                    for band, powers in band_powers.items():
                        record += band + "=" + str(round(powers[index], 2)) + ", "
                    record += "\n"
        neural_data[subj_id] = record
        if (subj_index % 20) == 0:
            print(subj_index, "subjects processed out of", len(bdf_index))
    return neural_data

def generate_neural_data(bdf_index):
    neural_data = {}
    for subj_index, (subj_id, bdf_paths) in enumerate(bdf_index.items()):
        record = {}
        for bdf_path in bdf_paths:
            band_powers, channel_names = process_bdf(bdf_path)
            if "restEC" in bdf_path.name:
                condition = "EC"
                #record += "Eyes Closed Spectral Features (μV²/Hz):\n"
            elif "restEO" in bdf_path.name:
                condition = "EO"
                #record += "Eyes Open Spectral Features (μV²/Hz):\n"
            else:
                print(subj_id)
                print(bdf_paths)
                continue
            
            condition_data = {}
            for index, channel in enumerate(channel_names):
                condition_data[channel] = {
                    band: round(powers[index], 2) for band, powers in band_powers.items()
                }
            record[condition] = condition_data

        neural_data[subj_id] = record
        if (subj_index % 20) == 0:
            print(subj_index, "subjects processed out of", len(bdf_index))
    return neural_data

def filter_df(participants_file):
    part_df = pd.read_excel(participants_file)

    neo_cols = [f'neoFFI_q{i}' for i in range(1, 61)]
    filt_df = part_df[
        (part_df["formal_status"] != "UNKNOWN") &
        (part_df["age"].notna()) &
        (part_df["gender"].notna()) &
        (part_df["education"].notna()) &
        (part_df["n_oddb_CP"].notna()) &
        (part_df["n_oddb_FP"].notna()) &
        (part_df["n_oddb_CN"].notna()) &
        (part_df["n_oddb_FN"].notna()) &
        (part_df["avg_rt_oddb_CP"].notna()) &
        (part_df[neo_cols].notna().all(axis=1))
    ]
    filt_df.drop_duplicates(subset="TDBRAIN_ID", keep="first", inplace=True)
    filt_df = filt_df.set_index("TDBRAIN_ID")
    return filt_df


def process_bdf(bdf_path):
    """Processes a single bdf path from end to end, outputting the structured text that the LLM will see."""
    # stores the raw data in raw_bdf
    raw_bdf = mne.io.read_raw_bdf(bdf_path, preload=True)
    # creates a copy (to preserve the original raw signal) and only selects the EEG channels
    bdf = raw_bdf.copy().pick('eeg')

    non_eeg = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'Mass']
    # some non eeg channels still remain after picking eeg due to a metadata mismatch, so this removes those as well
    bdf.drop_channels([ch for ch in non_eeg if ch in bdf.ch_names])
    # applies a bandpass filter (1-40 Hz) in order to remove movement artifacts and other noise
    bdf.filter(l_freq=1.0, h_freq=40.0)
    # transforms into signal power distributed across specific frequencies 
    spectrum = bdf.compute_psd(method='welch', fmin=1, fmax=40.0)
    # extract the psd data and the frequency ranges
    psds, freqs = spectrum.get_data(return_freqs=True)
    band_powers = {}
    # the frequency ranges and their corresponding band names
    freq_ranges = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 100.0)}
    for band, (fmin, fmax) in freq_ranges.items():
        # create a boolean mask for the freq range in question
        mask = (freqs >= fmin) & (freqs <= fmax)
        # in the psds data, keeps all dimensions same (through the ...), apply mask to the last dim (keeping only the freqs within the mask)
        # and then taskes the mean across that very last dimension (using -1)
        # finally, the multiplication by 1e12 converts these into microvolts squared to increase LLM comprehension with the numbers
        band_powers[band] = psds[..., mask].mean(axis=-1) * 1e12
    # returning not only the band powers but also the names of the channels
    return band_powers, bdf.ch_names

def sanitize(val):
    if isinstance(val, float) and math.isnan(val):
        return None
    return val