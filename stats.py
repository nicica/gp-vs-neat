import pandas as pd
import sys
import helpers.plots as plots


df = pd.read_csv('stats.csv')

if len(sys.argv) == 1:
    print(df)
elif len(sys.argv) == 2:
    if sys.argv[1] == "st":
        print(df)
    elif sys.argv[1] == "wr":
        plots.plot_pie_chart(df["Winner"].tolist(), "Winrate piechart")
    elif sys.argv[1] == "aet":
        plots.barplot(df["GP Average Evaluation Time"].tolist(), df["NEAT Average Evaluation Time"].tolist(),
                             "No. of simulation", "Average time of evaluetion (s)",
                               "Barplot of average evaluation time in each simulation")
    elif sys.argv[1] == "afs":
        plots.lineplot(df["GP Average Fitness Score"].tolist(), df["NEAT Average Fitness Score"],
                       "No. of simulation", "Average fitness score",
                       "Lineplot of average fitness score in each simulation")
    elif sys.argv[1] == "nbs":
        plots.barplot(df["GP Best Route Count"].tolist(), df["NEAT Best Route Count"].tolist(),
                             "No. of simulation", "No. of best paths found",
                               "Barplot of best paths found in each simulation")
    else:
        print("Invalid argument.")
else:
    print("Invalid number of arguments.")




