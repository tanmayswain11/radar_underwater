import numpy as np

def generate_signal():
    t = np.linspace(0, 1, 1000)

    # mix of frequencies (like radar)
    signal = (
        np.sin(2*np.pi*30*t) +
        0.6*np.sin(2*np.pi*80*t) +
        0.3*np.sin(2*np.pi*150*t)
    )

    # add noise
    noise = 0.2 * np.random.randn(len(t))
    signal = signal + noise

    return t, signal