import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def barplot(list1, list2, xlab, ylab, title, note = None):
    # Determine the longer list and pad the shorter one with NaN for alignment
    max_length = max(len(list1), len(list2))
    
    list1_extended = list1 + [np.nan] * (max_length - len(list1))
    list2_extended = list2 + [np.nan] * (max_length - len(list2))

    # Set positions for bars
    x_positions = np.arange(max_length)

    # Create the histogram
    bar_width = 0.4

    plt.bar(x_positions - bar_width / 2, list1_extended, width=bar_width, label='GP', color='blue', align='center')
    plt.bar(x_positions + bar_width / 2, list2_extended, width=bar_width, label='NEAT', color='orange', align='center')

    # Add labels and legend
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.xticks(x_positions, labels=range(1, max_length + 1))
    plt.legend()


    if note!=None:
        plt.text(0.78, 0.985, note, fontsize=10, color='black', transform=plt.gca().transAxes, va='top')

    plt.show()

def plot_scatter(list1, list2, xlab, ylab, title, lab2):
    
    x_values = range(1, len(list1) + 1)
    
    plt.scatter(x_values, list1, color='blue', label='Best score')
    plt.scatter(x_values, list2, color='orange', label=lab2)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(bbox_to_anchor=(0.26, 1.15))
    plt.grid(True, linestyle='--')
    plt.xticks(x_values)
    plt.show()

def plot_pie_chart(data, title):
    # Count the occurrences of each string in the list
    counter = Counter(data)
    
    # Prepare the data for the pie chart
    labels = counter.keys()
    sizes = counter.values()
    
    # Plot the pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.show()


def lineplot(list1, list2, xlab, ylab, title):
    
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    x_positions = range(1, len(list1) + 1)  # Generate x-axis positions

    plt.figure(figsize=(8, 6))

    # Plot the first list
    plt.plot(x_positions, list1, marker='o', label='GP', color='blue')
    # Plot the second list
    plt.plot(x_positions, list2, marker='o', label='NEAT', color='orange')

    plt.xticks(x_positions)  # Set x-ticks to match x positions
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

