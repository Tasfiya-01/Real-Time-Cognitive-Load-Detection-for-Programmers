import mne

path = r"C:\Users\HP\Downloads\CollectedData_2021-07-29_1.fif"

raw = mne.io.read_raw_fif(path, preload=False, verbose=False)

print(raw.info)
print("Sampling rate:", raw.info["sfreq"])
print("Duration (s):", raw.times[-1])
print("Channels:", raw.ch_names)
print("Annotations:", len(raw.annotations))
print(set(raw.annotations.description))
print(raw.annotations[:10])