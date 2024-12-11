# Co-evolution of Agents: Genetic Programming vs. Neuroevolution

This is a project for the course *Optimization for Artificial Intelligence* in my master's degree program. Its purpose is to demonstrate how agents evolve differently with Genetic Programming (GP) and Neuroevolution of Augmenting Topologies (NEAT). The core of this project (the environment) is based on a task I completed for my bachelor's degree course, which can be found in my other repository [q_learning_task](https://github.com/nicica/q-learning-task).

## A Quick Explanation

This simulation features a 7x12 grid representation of an environment where two agents compete against each other to find the best path to victory. First, they must collect five artifacts (tennis balls) and then find the path to the house while avoiding traps. One agent is developed using GP, while the other is developed using NEAT. The state of the map is represented using a reward matrix, which the evolutionary algorithms are modeled around.

After each simulation, several plots are generated:
- Best fitness score vs. the fitness score returned by GP in each evaluation.
- Same for NEAT in each evaluation.
- Growth of fitness scores through generations (comparison between GP and NEAT).
- Average evaluation time (also comparing GP and NEAT).

The information used for these plots, along with parameters like the number of generations, population sizes, and simulation outcomes, is saved in `stats.csv`.

## How to Run

The main simulation is implemented in `main.py`. Running this file (without any arguments) starts the simulation.

The `stats.py` file can be used to analyze the results of the simulations. It can be run without arguments or with specific arguments, as described below:

- **No arguments** or argument `st`: Outputs the `stats.csv` file as a dataframe.
- Argument `wr`: Generates a pie chart representing the win rates from the simulations.
- Argument `aet`: Produces a bar plot comparing the average evaluation times of GP and NEAT.
- Argument `afs`: Creates a line plot representing the average fitness scores of GP and NEAT across simulations.
- Argument `nbs`: Displays a bar plot showing how many times GP or NEAT found the best path in each simulation.
