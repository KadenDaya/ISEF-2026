import numpy as np

def one_hot_encode_gRNA(seq):
    mapping = {'A': [1,0,0,0], 'T': [0,1,0,0], 'C': [0,0,1,0], 'G': [0,0,0,1]}
    return np.array([mapping[base] for base in seq])

def scale_output(y_raw):
    return y_raw * 0.99 + 0.01

def unscale_output(y_scaled):
    return (y_scaled - 0.01) / 0.99

def encode_batch(seqs):
    return np.array([one_hot_encode_gRNA(seq) for seq in seqs])
