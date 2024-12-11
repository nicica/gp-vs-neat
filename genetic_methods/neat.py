from genetic_methods.network_properties import NeuronGene, LinkGene, NeuronType
import copy
from enviornment.environment import Action
import enviornment.artifacts as artifacts
import random
import enviornment.tiles as tiles
import numpy as np

class NEAT_Evaluator:
    def __init__(self, enviorment, score):
        self.p_m_i = 0

        self.original_map = [[enviorment.field_map[i][j].reward() for j in range(len(enviorment.field_map[i]))] 
    for i in range(len(enviorment.field_map))]

        self.population = []

        self.reward_map = copy.deepcopy(self.original_map)
        #adjust rewards based on where the opponent is
        self.adjust_for_oponent(enviorment.get_agent_position().copy(), score)

        
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
            chromosome = {
                "Neurons": [],
                "Links": [],
                "NextNeuronNumber" : 1
            }
            input_node = NeuronGene(enviorment, NeuronType(NeuronType.INPUT), current_score, chromosome["NextNeuronNumber"])
            chromosome["NextNeuronNumber"]+=1
            chromosome["Neurons"].append(input_node)
            for j in range(5):
                output_node = NeuronGene(enviorment, NeuronType(NeuronType.OUTPUT), current_score, chromosome["NextNeuronNumber"])
                chromosome["NextNeuronNumber"]+=1
                route = []
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
                    route.append(Action(ret_action))
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
                            #print("Ne: "+str(len(chromosome)))
                            break
                        elif enviorment.field_map[current_position[0]][current_position[1]].kind() == tiles.Hole.kind():
                           # print("Rupa: "+str(len(chromosome)))
                            break
                    else:
                        if enviorment.artifacts_map[artifacts.TennisBallA.kind()].get_position() == current_position:
                            break
                        elif enviorment.artifacts_map[artifacts.TennisBallB.kind()].get_position() == current_position:
                            break
                        elif enviorment.field_map[current_position[0]][current_position[1]].kind() == tiles.Hole.kind():
                            break
                link_gene = LinkGene(route, 1, j+2)
                chromosome["Links"].append(link_gene)
                route_score = self.fitness_score(route, starting_position)
                output_node.set_bias(route_score/100)
                output_node.set_value(link_gene.weight, input_node.value)
                chromosome["Neurons"].append(output_node)


            self.population.append(chromosome)
            NeuronGene.nextID = 0


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

    def get_best_route(self, chromosome):
        output_chromosomes = list(filter(lambda x: x.type==NeuronType(NeuronType.OUTPUT) ,chromosome["Neurons"]))
        ID_max = max(output_chromosomes, key= lambda x: x.value).ID

        for link in list(filter(lambda x: x.enabled, chromosome["Links"])):
            if link.destination_ID == ID_max:
                return link.route

    def selection(self, tournament_size, starting_position):
        tournament = random.choices(self.population, k=tournament_size)

        return max(tournament, key=lambda individual: self.fitness_score(self.get_best_route(individual), starting_position))

        
    def crossover(self, parent_a, parent_b):
        condition_1 = [nr.ID for nr in list(filter(lambda x: x.type==NeuronType(NeuronType.OUTPUT) ,parent_a["Neurons"]))] == [nr.ID for nr in list(filter(lambda x: x.type==NeuronType(NeuronType.OUTPUT) ,parent_b["Neurons"]))]
        condition_2 = [nr.ID for nr in list(filter(lambda x: x.type==NeuronType(NeuronType.HIDDEN) ,parent_a["Neurons"]))] == [nr.ID for nr in list(filter(lambda x: x.type==NeuronType(NeuronType.HIDDEN) ,parent_b["Neurons"]))]
        condition_3 = [(lnk.source_ID,lnk.destination_ID) for lnk in list(filter(lambda x: x.enabled ,parent_a["Links"]))] == [(lnk.source_ID,lnk.destination_ID) for lnk in list(filter(lambda x: x.enabled ,parent_b["Links"]))]
        if condition_1 and condition_2 and condition_3:
            links_a = list(filter(lambda x:x.source_ID==1 and x.enabled, parent_a['Links']))
            links_b = list(filter(lambda x:x.source_ID==1 and x.enabled, parent_b['Links']))
            offspring_a = {
                "Neurons": [],
                "Links": [],
                "NextNeuronNumber" : parent_a["NextNeuronNumber"]
            }
            offspring_b = {
                "Neurons": [],
                "Links": [],
                "NextNeuronNumber" : parent_b["NextNeuronNumber"]
            }
            pt_swp_b = int(len(links_a)/2)
            pt_swp_a = pt_swp_b + len(links_a)%2 
            #POPRAVITI OVAJ KROSOVER

            offspring_a["Neurons"].append(list(filter(lambda x:x.type==NeuronType(NeuronType.INPUT) ,parent_a["Neurons"]))[0])
            offspring_b["Neurons"].append(list(filter(lambda x:x.type==NeuronType(NeuronType.INPUT) ,parent_b["Neurons"]))[0])
            for i in range(len(links_a)):
                if i<pt_swp_a:
                    offspring_a["Links"].append(links_a[i])
                    linked_neuron = list(filter(lambda x:x.ID==links_a[i].destination_ID ,parent_a["Neurons"]))[0]
                    offspring_a["Neurons"].append(linked_neuron)
                    hidden_neurons = []
                    is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                    if is_hidden:
                        hidden_neurons.append(linked_neuron)
                    while len(hidden_neurons)>0:
                        hidden_neuron = hidden_neurons.pop(0)
                        new_links = list(filter(lambda x:x.source_ID==hidden_neuron.ID and x.enabled ,parent_a["Links"]))
                        for lk in new_links:
                            offspring_a["Links"].append(lk)
                            linked_neuron = list(filter(lambda x:x.ID==lk.destination_ID ,parent_a["Neurons"]))[0]
                            offspring_a["Neurons"].append(linked_neuron)
                            is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                            if is_hidden:
                                hidden_neurons.append(linked_neuron)
                    
                else:
                    offspring_b["Links"].append(links_a[i])
                    linked_neuron = list(filter(lambda x:x.ID==links_a[i].destination_ID ,parent_a["Neurons"]))[0]
                    offspring_b["Neurons"].append(linked_neuron)
                    hidden_neurons = []
                    is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                    if is_hidden:
                        hidden_neurons.append(linked_neuron)
                    while len(hidden_neurons)>0:
                        hidden_neuron = hidden_neurons.pop(0)
                        new_links = list(filter(lambda x:x.source_ID==hidden_neuron.ID and x.enabled ,parent_a["Links"]))
                        for lk in new_links:
                            offspring_b["Links"].append(lk)
                            linked_neuron = list(filter(lambda x:x.ID==lk.destination_ID ,parent_a["Neurons"]))[0]
                            offspring_b["Neurons"].append(linked_neuron)
                            is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                            if is_hidden:
                                hidden_neurons.append(linked_neuron)
                        
            for i in range(len(links_b)):
                if i>pt_swp_b:
                    offspring_a["Links"].append(links_b[i])
                    linked_neuron = list(filter(lambda x:x.ID==links_b[i].destination_ID ,parent_b["Neurons"]))[0]
                    offspring_a["Neurons"].append(linked_neuron)
                    hidden_neurons = []
                    is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                    if is_hidden:
                        hidden_neurons.append(linked_neuron)
                    while len(hidden_neurons)>0:
                        hidden_neuron = hidden_neurons.pop(0)
                        new_links = list(filter(lambda x:x.source_ID==hidden_neuron.ID and x.enabled ,parent_b["Links"]))
                        for lk in new_links:
                            offspring_a["Links"].append(lk)
                            linked_neuron = list(filter(lambda x:x.ID==lk.destination_ID ,parent_b["Neurons"]))[0]
                            offspring_a["Neurons"].append(linked_neuron)
                            is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                            if is_hidden:
                                hidden_neurons.append(linked_neuron)
                    
                else:
                    offspring_b["Links"].append(links_b[i])
                    linked_neuron = list(filter(lambda x:x.ID==links_b[i].destination_ID ,parent_b["Neurons"]))[0]
                    offspring_b["Neurons"].append(linked_neuron)
                    hidden_neurons = []
                    is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                    if is_hidden:
                        hidden_neurons.append(linked_neuron)
                    while len(hidden_neurons)>0:
                        hidden_neuron = hidden_neurons.pop(0)
                        new_links = list(filter(lambda x:x.source_ID==hidden_neuron.ID and x.enabled ,parent_b["Links"]))
                        for lk in new_links:
                            offspring_b["Links"].append(lk)
                            linked_neuron = list(filter(lambda x:x.ID==lk.destination_ID ,parent_b["Neurons"]))[0]
                            offspring_b["Neurons"].append(linked_neuron)
                            is_hidden = linked_neuron.type == NeuronType(NeuronType.HIDDEN)
                            if is_hidden:
                                hidden_neurons.append(linked_neuron)
                        
            return offspring_a, offspring_b
        else:
            return parent_a, parent_b        

    def get_good_output_ids(self, individual):
        routes_id_dest =[]
        start_points = [individual["Neurons"][0].ID]
        while len(start_points)>0:
            start_point = start_points.pop(0)
            batch = []
            for lks in list(filter(lambda x: x.source_ID == start_point and x.enabled, individual["Links"])):
                end_point = list(filter(lambda y: y.ID == lks.destination_ID, individual["Neurons"]))[0]
                if end_point.type == NeuronType(NeuronType.HIDDEN):
                    start_points.append(end_point.ID)
                if end_point.value >= 0.5:
                    batch.append(end_point.ID)
            routes_id_dest.append(batch) 

        return routes_id_dest
    
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

    def get_merging_point(self, routes, starting_position):
        sequences = []
        merging_routes = []
        original_routes = []
        best_position = None
        for route in routes:
            sequences.append(self.get_positions_sequence(route, starting_position))
        
        position_appearences = {}     
        #print(sequences)   
        for sq in sequences:

            for pos in set(tuple(inner_list) for inner_list in sq):
                if pos in position_appearences:
                    position_appearences[pos] += 1
                else:
                    position_appearences[pos] = 0
        max_value = max(position_appearences.values())

        if max_value>1:
            best_positions = [key for key, value in position_appearences.items() if value == max_value]

            
            if len(best_positions)>1:
                position_score = {}
                for pos in best_positions:
                    position_score[pos] = 0
                    for sq in sequences:
                         if pos in sequences:
                             position_score[pos] += abs(len(sq)/2 - sq.index(pos))
                best_position = min(position_score, key=position_score.get)
            else:
                best_position = best_positions[0]

            first_part_route = []
            for i in range(len(sequences)):
                if best_position in sequences[i]:
                    original_routes.append(routes[i])
                    merging_routes.append(routes[i][sequences[i].index(best_position)+1:])
                    first_part_route.append(routes[i][:sequences[i].index(best_position)+1])
            if len(first_part_route)>0:
                merging_routes.insert(0, max(first_part_route, key=lambda x: self.fitness_score(x, starting_position)))

        return merging_routes, original_routes, best_position


    def mutation(self, individual, p_m, starting_position, enviorment, current_score):
        # Create a copy of the individual
        new_individual = copy.deepcopy(individual)
        p_m += self.p_m_i
        
        # Update weights
        for i in range(len(new_individual["Links"])):
            if not new_individual["Links"][i].enabled:
                continue
            endpoint_neuron = list(filter(lambda x: new_individual["Links"][i].destination_ID == x.ID, new_individual["Neurons"]))[0]
            if endpoint_neuron.value >= 0.5:
                new_individual["Links"][i].increase_weight()
            else:
                new_individual["Links"][i].decrease_weight()

        routes_to_add = 0
        condition = False
        if random.random() < p_m:
            self.p_m_i = 0
            good_routes_mt = self.get_good_output_ids(new_individual)
            for good_routes in good_routes_mt:
                condition = len(good_routes) > 1
                if condition:
                    route_squences = []

                    for route in good_routes:
                        route_squences.append(list(filter(lambda x: x.enabled and x.destination_ID == route, new_individual["Links"]))[0].route)

                    merging_routes, original_routes, mpt = self.get_merging_point(route_squences, starting_position)

                    if len(merging_routes) == 0:
                        condition = False
                    else:
                        routes_to_add = len(merging_routes) - 2
                        hidden_node = NeuronGene(enviorment, NeuronType(NeuronType.HIDDEN), current_score, new_individual["NextNeuronNumber"])
                        new_individual["NextNeuronNumber"] += 1
                        hidden_node.set_bias(self.fitness_score(merging_routes[0], starting_position) / 100)
                        endpoints = []
                        startpoint = None
                        for org in original_routes:
                            for i in range(len(new_individual["Links"])):
                                if new_individual["Links"][i].route == org:
                                    new_individual["Links"][i].disable_link()
                                    endpoints.append(new_individual["Links"][i].destination_ID)
                                    startpoint = new_individual["Links"][i].source_ID
                        link_to_hidden = LinkGene(merging_routes[0], startpoint, hidden_node.ID)
                        new_individual["Links"].append(link_to_hidden)
                        new_individual["Neurons"].append(hidden_node)
                        i = 1
                        while i < len(merging_routes):
                            new_link = LinkGene(merging_routes[i], hidden_node.ID, endpoints[i - 1])
                            new_individual["Links"].append(new_link)
                            for j in range(len(new_individual["Neurons"])):
                                if new_individual["Neurons"][j].ID == endpoints[i - 1]:
                                    new_individual["Neurons"][j].set_bias(self.fitness_score(merging_routes[i], mpt) / 100)
                                    break
                            i += 1
                        break

            if not condition:
                routes_to_add = 1
                output_chromosomes = list(filter(lambda x: x.type == NeuronType(NeuronType.OUTPUT), new_individual["Neurons"]))
                node_min = min(output_chromosomes, key=lambda x: x.value)
                new_individual["Neurons"].remove(node_min)
                for i in range(len(new_individual["Links"])):
                    if new_individual["Links"][i].destination_ID == node_min.ID:
                        new_individual["Links"][i].disable_link()
                        break
        else:
            self.p_m_i += abs(np.random.normal(loc=0,scale=1))

        for i in range(routes_to_add):
            new_node = NeuronGene(enviorment, NeuronType.OUTPUT, current_score, new_individual["NextNeuronNumber"])
            new_individual["NextNeuronNumber"] += 1
            route = []
            current_position = []
            current_position.append(starting_position[0])
            current_position.append(starting_position[1])
            while True:
                while True:
                    ret_action = random.randint(0, 3)
                    if ret_action == 0 and current_position[0] == 0:
                        continue
                    elif ret_action == 1 and current_position[1] == 0:
                        continue
                    elif ret_action == 2 and current_position[0] == 6:
                        continue
                    elif ret_action == 3 and current_position[1] == 11:
                        continue
                    break
                route.append(Action(ret_action))
                if ret_action == 0:
                    current_position[0] -= 1
                elif ret_action == 2:
                    current_position[0] += 1
                elif ret_action == 1:
                    current_position[1] -= 1
                elif ret_action == 3:
                    current_position[1] += 1
                if current_score >= 5:
                    if enviorment.artifacts_map[artifacts.Goal.kind()].get_position() == current_position:
                        break
                    elif enviorment.field_map[current_position[0]][current_position[1]].kind() == tiles.Hole.kind():
                        break
                else:
                    if enviorment.artifacts_map[artifacts.TennisBallA.kind()].get_position() == current_position:
                        break
                    elif enviorment.artifacts_map[artifacts.TennisBallB.kind()].get_position() == current_position:
                        break
                    elif enviorment.field_map[current_position[0]][current_position[1]].kind() == tiles.Hole.kind():
                        break
            new_link_1 = LinkGene(route, 1, new_node.ID)
            new_node.set_bias(self.fitness_score(route, starting_position) / 100)
            new_individual["Neurons"].append(new_node)
            new_individual["Links"].append(new_link_1)

        return new_individual
    
    def get_best(self, starting_position):
        if not self.population:
            raise ValueError("Population is empty.")
    
        valid_individuals = [(self.fitness_score(self.get_best_route(ind), starting_position), self.get_best_route(ind)) for ind in self.population]

        if not valid_individuals:
            raise ValueError("No valid individuals in the population.")

        return max(valid_individuals, key=lambda x: x[0])

    def generation(self, starting_position, p_m, elitism, enviorment, current_score):
        new_population = []

        # If elitism is enabled, preserve the top `elitism` individuals
        if elitism > 0:
            elites = sorted(
                self.population, 
                key=lambda ind: self.fitness_score(self.get_best_route(ind), starting_position), 
                reverse=True
            )[:elitism]
            new_population.extend(elites)

        # Generate the remaining population through selection, crossover, and mutation
        while len(new_population) < len(self.population):
            parent_a = self.selection(4, starting_position)  # Tournament size = 3
            parent_b = self.selection(4, starting_position)

            offspring_a, offspring_b = self.crossover(parent_a, parent_b)
            
            offspring_a = self.mutation(offspring_a, p_m, starting_position, enviorment, current_score)
            offspring_b = self.mutation(offspring_b, p_m, starting_position, enviorment, current_score)

            new_population.append(offspring_a)
            if len(new_population) < len(self.population):
                new_population.append(offspring_b)

        # Replace the old population with the new population
        self.population = new_population

    def update_values(self):
        for ind in self.population:
            for i in range(len(ind["Neurons"])):
                if ind["Neurons"][i].type == NeuronType(NeuronType.INPUT):
                    continue
                link = list(filter(lambda x: x.destination_ID == ind["Neurons"][i].ID and x.enabled, ind["Links"]))[0]
                input_nr = list(filter(lambda x: x.ID == link.source_ID, ind["Neurons"]))[0]
                ind["Neurons"][i].set_value(link.weight, input_nr.value)

    def evaluate(self, population_size, starting_position, current_score, environment, n_gen, p_m, elitism):
        self.init_population(population_size, starting_position, current_score, environment)
        first_fitness = 0
        for generation_number in range(n_gen):
            self.generation(starting_position, p_m, elitism, environment, current_score)
            self.update_values()
            best_fitness, _ = self.get_best(starting_position)
            if generation_number == 0:
                first_fitness = best_fitness
           # print(f"NEAT Generation {generation_number + 1}/{n_gen}: Best Fitness = {best_fitness} ")

        # Return the best individual from the final population
        best_fitness, best_individual = self.get_best(starting_position)
        #print(f"NEAT Best fitness: {best_fitness}")
        return best_individual, best_fitness, first_fitness

    def print_pop(self, population):
        #provera
        p = 1
        for pop in population:
            print("========================================")
            print(f"Ch {p}")
            p+=1
            print("Neurons:")
            for nd in pop["Neurons"]:
                print(f"ID: {nd.ID}")
                print(f"Type: {nd.type}")
                print(f"Value: {nd.value}")
                print(f"Bias: {nd.bias}")
            print("Links:")
            for nd in pop["Links"]:
               # print(f"Reoute: {nd.route}")
                print(f"Enabled: {nd.enabled}")
                print(f"Src: {nd.source_ID}")
                print(f"Dst: {nd.destination_ID}")
                print(f"Weight: {nd.weight}")