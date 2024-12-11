
from pprint import pprint
import copy
import random
from enviornment.environment import Action
import enviornment.tiles as tiles
import enviornment.artifacts as artifacts

class GP_Evaluator:

    def __init__(self, enviorment, score):
        self.original_map = [[enviorment.field_map[i][j].reward() for j in range(len(enviorment.field_map[i]))] 
    for i in range(len(enviorment.field_map))]

        self.population = []

        self.reward_map = copy.deepcopy(self.original_map)
        #adjust rewards based on where the opponent is
        self.adjust_for_oponent(enviorment.get_robot_position().copy(), score)

        
    def adjust_for_oponent(self, oponent_position, score):
        adjusment_level = -200
        self.reward_map[oponent_position[0]][oponent_position[1]] = -50
        for i in range(1, 5):
            positions_to_be_changed = []
        
            for j in range(0, 4 * i):
                new_position = oponent_position[:]
                
                if j.bit_length() <= 2:
                    new_position[j%2] += i if (j>>1) % 2 == 0 else -i
                elif j.bit_length() == 3:
                    new_position[0] += (i-1) if (j>>1) % 2 == 0 else (-i+1)
                    new_position[1] += 1 if j%2 == 0 else -1
                else:
                    if j>7 and j<12:
                        new_position[1] += (i-1) if (j>>1) % 2 == 0 else (-i+1)
                        new_position[0] += 1 if j%2 == 0 else -1
                    else:
                        new_position[0] += 2 if (j>>1) % 2 == 0 else -2
                        new_position[1] += 2 if j%2 == 0 else -2
               
                positions_to_be_changed.append(new_position)

            positions_to_be_changed = [el for el in positions_to_be_changed if 0 <= el[0] <= 6 and 0 <= el[1] <= 11]

            for el in positions_to_be_changed:
                self.reward_map[el[0]][el[1]] = -1000 * (score+1) if self.original_map[el[0]][el[1]]==-1000 else (adjusment_level * 10 if self.original_map[el[0]][el[1]] == -10 else adjusment_level)

            adjusment_level/=4
                
    def init_population(self, population_size, starting_position, current_score, enviorment):
        self.population = []
        for i in range(population_size):
            chromosome = []
            current_position= []
            current_position.append(starting_position[0])
            current_position.append(starting_position[1])
            while True:
                while True:
                    ret_action = random.randint(0,3)
                    if ret_action == 0 and current_position[0]==0:
                        continue
                    elif ret_action == 1 and current_position[1]==0:
                        continue
                    elif ret_action == 2 and current_position[0]==6:
                        continue
                    elif ret_action == 3 and current_position[1]==11:
                        continue
                    break
                chromosome.append(Action(ret_action))
                if ret_action == 0:
                    current_position[0] -=1
                elif ret_action == 2:
                    current_position[0] +=1
                elif ret_action == 1:
                    current_position[1] -=1
                elif ret_action == 3:
                    current_position[1] +=1
                if current_score>=5:
                    if enviorment.artifacts_map[artifacts.Goal.kind()].get_position() == current_position:
                       # print("Ne: "+str(len(chromosome)))
                        break
                    elif enviorment.field_map[current_position[0]][current_position[1]].kind() == tiles.Hole.kind():
                      #  print("Rupa: "+str(len(chromosome)))
                        break
                else:
                    if enviorment.artifacts_map[artifacts.TennisBallA.kind()].get_position() == current_position:
                        break
                    elif enviorment.artifacts_map[artifacts.TennisBallB.kind()].get_position() == current_position:
                        break
                    elif enviorment.field_map[current_position[0]][current_position[1]].kind() == tiles.Hole.kind():
                        break
            self.population.append(chromosome)
                

    def fitness_score(self, actions, starting_position):
        score = 0
        position = []
        position.append(starting_position[0])
        position.append(starting_position[1])
        for act in actions:
            if act == Action(0):
                position[0] -=1
            elif act == Action(2):
                position[0] +=1
            elif act == Action(1):
                position[1] -=1
            elif act == Action(3):
                position[1] +=1
            score += self.reward_map[position[0]][position[1]]
        return score

    def selection(self, tournament_size, starting_position):
        tournament = random.choices(self.population, k=tournament_size)
    
        return max(tournament, key=lambda individual: self.fitness_score(individual, starting_position))
    
    def get_positions_sequence(self, actions, starting_position):
        # Simulate actions and track positions visited
        positions = []  # Start with the initial position
        position = []
        position.append(starting_position[0])
        position.append(starting_position[1])
        
        
        for act in actions:
            if act == Action(0):  # UP
                position[0] -= 1
            elif act == Action(2):  # DOWN
                position[0] += 1
            elif act == Action(1):  # LEFT
                position[1] -= 1
            elif act == Action(3):  # RIGHT
                position[1] += 1
            positions.append(position[:])  # Add current position to track
        return positions
    

    #prvo odluciti koji krosover
    def find_crossover_points(self, pos_seq_a, pos_seq_b):
        matching_points = []
        # Identify matching positions that aren't the first or last
        continue_next = False
        b_sek = 1
        for i in range(1, len(pos_seq_a) - 2):
            if continue_next:
                continue_next = False
                continue
            for j in range(b_sek, len(pos_seq_b) - 2):
                if pos_seq_a[i] == pos_seq_b[j]:  # Matching positions
                    matching_points.append((i, j))
                    continue_next = True
                    b_sek = j+1 if j< (len(pos_seq_b) - 2) else 1
                    break
        
        # Ensure we have at least two distinct matching pairs
        if len(matching_points) >= 2:
            return matching_points[0], matching_points[1]  # Start and end points
        else:
            return None  # No valid crossover points found

    def crossover(self, parent_a, parent_b, starting_position):
        # Get positions sequence for both parents
        pos_seq_a = self.get_positions_sequence(parent_a, starting_position)
        pos_seq_b = self.get_positions_sequence(parent_b, starting_position)

        # Find valid crossover points
        crossover_points = self.find_crossover_points(pos_seq_a, pos_seq_b)
        if not crossover_points:
            return parent_a, parent_b  # No crossover possible, return parents as is

        (start_a, start_b), (end_a, end_b) = crossover_points

        # Create offspring by swapping the segments
        offspring_a = parent_a[:(start_a+1)] + parent_b[(start_b+1):(end_b+1)] + parent_a[(end_a+1):]
        offspring_b = parent_b[:(start_b+1)] + parent_a[(start_a+1):(end_a+1)] + parent_b[(end_b+1):]
        
        return offspring_a, offspring_b
    
    def new_end_of_route(self, last_move, last_position, enviorment):
        """ Adjusts the last move if it leads to a Hole, and generates moves until a valid terminal is reached. """
        position = last_position[:]
        moves = []
        possible_moves = [Action(0), Action(1), Action(2), Action(3)]  # UP, LEFT, DOWN, RIGHT
        random.shuffle(possible_moves)
        for move in possible_moves:
            new_position = self.update_position(position, move)
            if self.is_legal_position(new_position) and move!=last_move:
                moves.append(move)
                position = new_position[:]
                break
        i = 0
        while enviorment.field_map[position[0]][position[1]].kind() != tiles.Hole.kind() or enviorment.artifacts_map[artifacts.TennisBallA.kind()].get_position() != position or enviorment.artifacts_map[artifacts.TennisBallB.kind()].get_position() != position:
            possible_moves = [Action(0), Action(1), Action(2), Action(3)] 
            random.shuffle(possible_moves)
            for move in possible_moves:
                new_position = self.update_position(position, move)
                if self.is_legal_position(new_position):
                    moves.append(move)
                    position = new_position[:]
                    break
            if i>50:
                return [last_move]
            i += 1

        return moves  # New moves that avoid a Hole

    def new_random_route(self, point_a, point_b, enviorment):
        """ Generate a random sequence of legal moves from point A to point B, avoiding Holes. """
        position = point_a[:]
        moves = []
        
        while position != point_b:
            possible_moves = [Action(0), Action(1), Action(2), Action(3)]  # UP, LEFT, DOWN, RIGHT
            random.shuffle(possible_moves)
            for move in possible_moves:
                new_position = self.update_position(position, move)
                if self.is_legal_position(new_position) and enviorment.field_map[new_position[0]][new_position[1]].kind() != tiles.Hole.kind():
                    moves.append(move)
                    position = new_position[:]
                    break

        return moves  # Random sequence of moves to reach point B

    def mutate(self, starting_position, individual, p_m, environment):
        """ Mutation process based on the individual's terminal position and mutation probability p_m. """
        position = starting_position[:]
        end_position = self.get_positions_sequence(individual, starting_position)[-1]

        if random.random() < p_m:
            # 1. Check if terminal position is a Hole
            if environment.field_map[end_position[0]][end_position[1]].kind() == tiles.Hole.kind():
                # If so, change the last move and extend the route with new moves to avoid Holes
                #return individual
                individual[-1:] = self.new_end_of_route(individual[-1], end_position, environment)
        
            # 2. Otherwise, replace a random sub-sequence with a new random route
            else:
                pos_sequence = self.get_positions_sequence(individual, starting_position)
                # Select two random points a and b that are separated by at least one move
                if len(pos_sequence) >= 5:
                    a = random.randint(0, int(len(pos_sequence)/2))
                    b = random.randint(a+2, len(pos_sequence)-1)
                    new_part = self.new_random_route(pos_sequence[a], pos_sequence[b], environment)
                    individual = individual[:(a+1)] + new_part+ individual[(b+1):]

        return individual  # Mutated individual

    def update_position(self, position, move):
        """ Update position based on move action. """
        pos = position[:]
        if move == Action(0):  # UP
            pos[0] -= 1
        elif move == Action(2):  # DOWN
            pos[0] += 1
        elif move == Action(1):  # LEFT
            pos[1] -= 1
        elif move == Action(3):  # RIGHT
            pos[1] += 1
        return pos

    def is_legal_position(self, position):
        """ Checks if a position is within the grid bounds and legal. """
        x, y = position
        return 0 <= x < 7 and 0 <= y < 12

    def get_best(self, starting_position):
        if not self.population:
            raise ValueError("Population is empty.")
    
        valid_individuals = [(self.fitness_score(ind, starting_position), ind) for ind in self.population 
                         if isinstance(ind, list) and all(isinstance(action, Action) for action in ind)]

        if not valid_individuals:
            raise ValueError("No valid individuals in the population.")

        return max(valid_individuals, key=lambda x: x[0]) 

    def generation(self, starting_position, p_m, elitism, enviorment):
        new_population = []

        # If elitism is enabled, preserve the top `elitism` individuals
        if elitism > 0:
            elites = sorted(
                self.population, 
                key=lambda ind: self.fitness_score(ind, starting_position), 
                reverse=True
            )[:elitism]
            new_population.extend(elites)

        # Generate the remaining population through selection, crossover, and mutation
        while len(new_population) < len(self.population):
            parent_a = self.selection(5, starting_position)  # Tournament size = 3
            parent_b = self.selection(5, starting_position)

            offspring_a, offspring_b = self.crossover(parent_a, parent_b, starting_position)

            offspring_a = self.mutate(starting_position, offspring_a, p_m, environment=enviorment)
            offspring_b = self.mutate(starting_position, offspring_b, p_m, environment=enviorment)

            new_population.append(offspring_a)
            if len(new_population) < len(self.population):
                new_population.append(offspring_b)

        # Replace the old population with the new population
        self.population = new_population

    def GP(self, population_size, starting_position, current_score, environment, n_gen, p_m, elitism):
        self.init_population(population_size, starting_position, current_score, environment)
        first_fitness = 0
        for generation_number in range(n_gen):
            self.generation(starting_position, p_m, elitism, environment)
            best_fitness, _ = self.get_best(starting_position)
            if generation_number == 0:
                first_fitness = best_fitness
           # print(f"GP Generation {generation_number + 1}/{n_gen}: Best Fitness = {best_fitness} ")

        # Return the best individual from the final population
        best_fitness, best_individual = self.get_best(starting_position)
        #print(f"GP Best fitness: {best_fitness}")
        return best_individual, best_fitness, first_fitness

    def print_reward_map(self):
        pprint(self.reward_map)