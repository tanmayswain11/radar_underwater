import os
import matplotlib.pyplot as plt

def plot_distribution(path):
    classes = os.listdir(path)
    counts = [len(os.listdir(f"{path}/{c}")) for c in classes]

    fig, ax = plt.subplots(figsize=(5, 4.5))  # ✅ size here
    ax.bar(classes, counts)

    ax.set_title("Dataset")
    return fig