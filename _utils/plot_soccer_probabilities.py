import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_soccer_probabilities(home_prob: np.array,
                              draw_prob: np.array,
                              away_prob: np.array,
                              type="density"):

    # Create the plot figure
    plt.figure(figsize=(10, 6))  # Größe des Plots festlegen

    if type=="density":
        sns.kdeplot(home_prob, color='green', label='Home', fill=True)
        sns.kdeplot(draw_prob, color='blue', label='Draw', fill=True)
        sns.kdeplot(away_prob, color='red', label='Away', fill=True)
    elif type=="histogram":
        plt.hist(home_prob, bins=100, color='blue', alpha=0.5, label='Home')
        plt.hist(draw_prob, bins=100, color='green', alpha=0.5, label='Draw')
        plt.hist(away_prob, bins=100, color='red', alpha=0.5, label='Away')

    # Achsenbeschriftungen und Titel
    plt.xlabel('Probability')
    plt.ylabel('Desity/N')
    plt.title('Probability Distribution')

    # Legende anzeigen
    plt.legend()

    # Plot anzeigen
    plt.show()