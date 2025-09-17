# visualizer.py
import matplotlib.pyplot as plt

def plot_results(y_true, y_pred, title):
    """Plot true vs. predicted scatter."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.show()
