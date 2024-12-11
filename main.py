import random
import enviornment.config as config
import pandas as pd
from enviornment.artifacts import Goal, Agent, Robot, TennisBallA, TennisBallB
from enviornment.environment import Environment, Quit, Action
import time
import threading
import enviornment.tiles as tiles
import tkinter as tk
from genetic_methods.gp import GP_Evaluator
from genetic_methods.neat import NEAT_Evaluator
import time
import helpers.dijkstra as dijkstra
import helpers.plots as plots

n_gen = 20

gp_pop_size = 50
neat_pop_size = 35

winner = None

environment = Environment(f'maps/map.txt')
dict_for_moves = {
    Action.UP: "Up",
    Action.LEFT: "Left",
    Action.DOWN: "Down",
    Action.RIGHT: "Right"
}

duzina = len(environment.field_map)
sirina = len(environment.field_map[0])

next_act = {
    "r": 0,
    "a": 0
}

actions = {
    "r": [],
    "a": []
}

next_positions = {
            "r": environment.get_robot_position().copy(),
            "a": environment.get_agent_position().copy()
}

scores = {
    "r": 0,
    "a": 0
}

penalties = {
    "r": 0,
    "a": 0
}
game_end = {
    "r": False,
    "a": False
}
evaluating = {
    "r": True,
    "a": True
}

time_per_evaluation = {
    'r': [],
    'a': []
}

best_fit = {
    'r': [],
    'a': []
}

actual_fit = {
    'r': [],
    'a': []
}

growth = {
    'r': [],
    'a': []
}


destroy_windows = False

def agent_evaluating(start_pos):
    start_time = time.time()
    global environment
    global evaluating
    global next_act
    gp_evaluator = GP_Evaluator(environment, scores["a"])

    best_actions, best_f, first_f = gp_evaluator.GP(population_size=gp_pop_size,
        starting_position=start_pos,
        current_score=scores['a'],
        environment=environment,
        n_gen=n_gen,
        p_m=0.2,
        elitism=2)
    
    evaluating["a"] = False

    actions["a"] = best_actions
    next_act["a"] = 0

    end_time = time.time()

    time_per_evaluation["a"].append(end_time-start_time)
    if scores["a"] < 5:
        best_fit["a"].append(max([
            dijkstra.get_best_score(gp_evaluator.reward_map, start_pos, environment.artifacts_map[TennisBallA.kind()].get_position()),
            dijkstra.get_best_score(gp_evaluator.reward_map, start_pos, environment.artifacts_map[TennisBallB.kind()].get_position())
            ]))
    else:
        best_fit["a"].append(
            dijkstra.get_best_score(gp_evaluator.reward_map, start_pos, environment.artifacts_map[Goal.kind()].get_position())
            )
    growth["a"].append(abs(best_f-first_f))
    actual_fit["a"].append(best_f)
    time.sleep(0.5)


def robot_evaluating(start_pos):
    start_time = time.time()

    global environment
    global evaluating
    global next_act
    neat_evaluator = NEAT_Evaluator(environment, scores["r"])

    best_actions,best_f, first_f = neat_evaluator.evaluate(population_size=neat_pop_size,
        starting_position=start_pos,
        current_score=scores['r'],
        environment=environment,
        n_gen=n_gen,
        p_m=0.2,
        elitism=2)
    
    evaluating["r"] = False

    actions["r"] = best_actions
    next_act["r"] = 0
    
    end_time = time.time()

    time_per_evaluation["r"].append(end_time-start_time)
    if scores["r"] < 5:
        best_fit["r"].append(max([
            dijkstra.get_best_score(neat_evaluator.reward_map, start_pos, environment.artifacts_map[TennisBallA.kind()].get_position()),
            dijkstra.get_best_score(neat_evaluator.reward_map, start_pos, environment.artifacts_map[TennisBallB.kind()].get_position())
            ]))
    else:
        best_fit["r"].append(
            dijkstra.get_best_score(neat_evaluator.reward_map, start_pos, environment.artifacts_map[Goal.kind()].get_position())
            )
    actual_fit["r"].append(best_f)
    growth["r"].append(abs(best_f-first_f))
    time.sleep(0.5)

    
def reevaluate_choice_agent():
    global actions
    global next_act

    if next_act["a"] == len(actions["a"]):
        return True
    
    fact = 1 if (len(actions["a"])/2 - next_act["a"]) == 0 else (len(actions["a"])/2 - next_act["a"])
    base = (0.5 + 0.4/fact) * ((next_act["a"])/8)

    return base > random.random()

def reevaluate_choice_robot():
    global actions
    global next_act

    if next_act["r"] == len(actions["r"]):
        return True
    
    fact = 1 if (len(actions["r"])/2 - next_act["r"]) == 0 else (len(actions["r"])/2 - next_act["r"])
    base = (0.5 + 0.4/fact) * ((next_act["r"])/8)

    return base > random.random()

def gp_action(agent_position, kind):
    global next_act
    global actions
    if environment.field_map[agent_position[0]][agent_position[1]].kind() == tiles.Hole.kind():
            environment.reset(kind)
            next_positions[kind]=environment.artifacts_map[kind].get_position().copy()
            scores[kind]=0
    else:    
        if penalties[kind] == 0:
            ret_action = actions["a"][next_act["a"]]
            next_act["a"] += 1
            next_positions[kind], _, game_end[kind], scores[kind] = environment.step(Action(ret_action), kind, scores[kind])
            if reevaluate_choice_agent():
                evaluating["a"] = True
            if environment.field_map[next_positions[kind][0]][next_positions[kind][1]].kind() == tiles.Mud.kind():
                penalties[kind] += random.randint(1,3)
        else:
            penalties[kind] -= 1
    
    time.sleep(0.5)


def neat_action(agent_position, kind):
    global next_act
    global actions
    if environment.field_map[agent_position[0]][agent_position[1]].kind() == tiles.Hole.kind():
            environment.reset(kind)
            next_positions[kind]=environment.artifacts_map[kind].get_position().copy()
            scores[kind]=0
    else:    
        if penalties[kind] == 0:
            ret_action = actions["r"][next_act["r"]]
            next_act["r"] += 1
            next_positions[kind], _, game_end[kind], scores[kind] = environment.step(Action(ret_action), kind, scores[kind])
            if reevaluate_choice_robot():
                evaluating["r"] = True
            if environment.field_map[next_positions[kind][0]][next_positions[kind][1]].kind() == tiles.Mud.kind():
                penalties[kind] += random.randint(1,3)
        else:
            penalties[kind] -= 1
    
    time.sleep(0.5)


def simulate_parallel():
    global destroy_windows
    global winner
    try:
        scores_thread = threading.Thread(target=scores_thread_f)
        scores_thread.start()
        environment.reset(Agent.kind())
        environment.reset(Robot.kind())
        position_robot= environment.get_robot_position().copy()
        position_agent= environment.get_agent_position().copy()
        environment.render(config.FPS)
        while True:
            args_agent = [position_agent] if evaluating["a"] else [position_agent, Agent.kind()]
            target_agent = agent_evaluating if evaluating["a"] else gp_action

            args_robot = [position_robot] if evaluating["r"] else [position_robot, Robot.kind()]
            target_robot = robot_evaluating if evaluating["r"] else neat_action


            agent_thread = threading.Thread(target=target_agent, args=args_agent)
            robot_thread = threading.Thread(target=target_robot, args=args_robot)

            agent_thread.start()
            robot_thread.start()

            agent_thread.join()
            robot_thread.join()
            position_robot = next_positions['r']
            position_agent = next_positions['a']

            if position_agent == position_robot:
                scores['a'] = scores['a'] - 1 if scores["a"]>0 else scores['a']
                scores['r'] = scores['r'] - 1 if scores["r"]>0 else scores['r']
                game_end["r"] = game_end["a"] = False

            environment.render(config.FPS)
            if game_end["r"] or game_end["a"]:
                if game_end["r"]:
                    winner = "NEAT"
                elif game_end['a']:
                    winner = "GP"
                destroy_windows = True
                scores_thread.join()
                break
    except Quit:
        pass


def update_label(window, label):
    # Update the label text with the latest scores
    label.config(text=f"Doggo: {scores['a']}, RoboDoggo: {scores['r']}")
    if destroy_windows:
        window.destroy()
    else:
        label.after(1000, update_label,window, label)

def scores_thread_f():
    window = tk.Tk()
    window.title("Scores")
    label = tk.Label(window, text=f"Doggo: {scores['a']}, RoboDoggo: {scores['r']}")
    label.pack(pady=20)

    update_label(window ,label)

    window.mainloop()



simulate_parallel()

plots.plot_scatter(best_fit["a"],actual_fit["a"], "No. of evaluation", "Fitness score", "Fitness score: GP vs Actual", "GP")
plots.plot_scatter(best_fit["r"],actual_fit["r"], "No. of evaluation", "Fitness score", "Fitness score: NEAT vs Actual", "NEAT")
plots.barplot(growth["a"],growth["r"],"No. of evaluation", "Growth per evaluation" ,
                      "Barplot of how fitness score has grown through generations on each evaluation")

note  = f"Average time\nper evaluation:\nGP: {sum(time_per_evaluation["a"])/len(time_per_evaluation["a"]):.2f}\nNEAT: {sum(time_per_evaluation["r"])/len(time_per_evaluation["r"]):.2f}"
plots.barplot(time_per_evaluation["a"],time_per_evaluation["r"],"No. of evaluation", "Time taken per evaluation (s)" ,
                      "Barplot of how much time each evaluation took", note)

gp_best_route_count = 0
neat_best_route_count = 0

for i in range(len(best_fit["a"])):
    if best_fit["a"][i]-actual_fit['a'][i] == 0:
        gp_best_route_count += 1

for i in range(len(best_fit["r"])):
    if best_fit["r"][i]-actual_fit['r'][i] == 0:
        neat_best_route_count += 1

df = pd.read_csv('stats.csv')

simulation_stats = {
    'n_gen':n_gen,
    'GP pop_size':gp_pop_size,
    'NEAT pop_size':neat_pop_size,
    'GP Average Fitness Score':round(sum(actual_fit["a"])/len(actual_fit["a"]),2),
    'NEAT Average Fitness Score':round(sum(actual_fit["r"])/len(actual_fit["r"]),2),
    'GP Average Evaluation Time':round(sum(time_per_evaluation["a"])/len(time_per_evaluation["a"]),2),
    'NEAT Average Evaluation Time':round(sum(time_per_evaluation["r"])/len(time_per_evaluation["r"]),2),
    'GP Number of Evaluations':len(time_per_evaluation["a"]),
    'NEAT Number of Evaluations':len(time_per_evaluation["r"]),
    'GP Best Route Count':gp_best_route_count,
    'NEAT Best Route Count':neat_best_route_count,
    'Winner': winner
}

df.loc[len(df)] = simulation_stats

df.to_csv('stats.csv',index=False)



