from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract(ppg, fs):
    # filter
    fn = fs / 2
    fl = 0.5
    fh = 10
    window_peak = int(fs * 0.111)
    window_beat = int(fs * 0.667)
    beta = 0.02

    b, a = signal.cheby2(2, 20, [fl / fn, fh / fn], 'bandpass')
    ppg_filtered = signal.filtfilt(b, a, ppg)

    ppg_clipped = ppg_filtered.copy()
    ppg_clipped[ppg_clipped < 0] = 0  #clipping
    ppg_square = ppg_clipped * ppg_clipped  # square

    # peak average
    mean_peak = pd.Series(ppg_square).rolling(window=window_peak).mean()
    for index in range(window_peak):
        mean_peak[index] = sum(ppg_square[0:index]) / (index + 1)
    # beat average
    mean_beat = pd.Series(ppg_square).rolling(window=window_beat).mean()
    for index in range(window_beat):
        mean_beat[index] = sum(ppg_square[0:index]) / (index + 1)
    z = np.mean(ppg_square)
    # find the block of interest
    alpha = beta * z
    thr1 = mean_beat + alpha

    block_of_interest = np.zeros(len(mean_peak), dtype=np.int)
    block_onset = []
    block_end = []
    for index in range(len(block_of_interest)):
        if mean_peak[index] > thr1[index]:
            block_of_interest[index] = 1
            if index > 0:
                if mean_peak[index-1] <= thr1[index-1]:
                    block_onset.append(index)
        else:
            if len(block_onset) > 0:
                if mean_peak[index-1] > thr1[index-1]:
                    block_end.append(index)

    peaks = []
    for index in range(len(block_end)):
        if block_end[index] - block_onset[index] > window_peak:
            segment = ppg_square[block_onset[index]:block_end[index]]
            peaks.append(np.argmax(segment) + block_onset[index])

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(ppg_filtered)
    # ax1.plot(peaks, ppg_filtered[peaks], 'r*')
    # ax2 = fig.add_subplot(212, sharex=ax1)
    # ax2.plot(ppg_square)
    # ax2.plot(mean_peak)
    # ax2.plot(mean_beat)
    # ax2.plot(thr1)
    # ax2.plot(block_of_interest)
    # ax2.plot(peaks, ppg_square[peaks], 'r*')
    return peaks




