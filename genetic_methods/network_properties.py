from enum import Enum
import numpy as np

class NeuronType(Enum):
    INPUT = 0
    OUTPUT = 2
    HIDDEN = 1

class NeuronGene:
    

    def __init__(self, enviorment, type, score, ID):
        self.ID = ID
        self.type = type
        self.value = None
        self.bias = None
        if type == NeuronType(NeuronType.INPUT):
            dist = 0
            if score<5:
                dist = min(enviorment.get_euklid_distance_tennis_ball_a(), enviorment.get_euklid_distance_tennis_ball_b())
            else:
                dist = enviorment.get_euklid_distance_house()
            self.value = (20-dist)/19 
        pass

    def set_bias(self, bias):
        if self.type == NeuronType(NeuronType.INPUT):
            raise ValueError("Can only set bias for output or hidden nodes")
        else:
            self.bias = bias

    def set_value(self, weight, input_node_value):
        if self.type == NeuronType(NeuronType.INPUT):
            raise ValueError("Can only set value for output or hidden nodes")
        else:
            z = weight * input_node_value + self.bias
            z = np.clip(z, -500, 500)
            self.value = 1 / (1 + np.exp(-z))

    def get_id(self):
        return self.ID
    
    @staticmethod
    def reset_ID():
        nextID = 0

class LinkGene:
    def __init__(self, route, src, dst):
        self.route = route
        self.weight = np.random.normal(loc=0, scale=1)
        self.enabled = True
        self.source_ID = src
        self.destination_ID = dst
    
    def increase_weight(self):
        self.weight += abs(np.random.normal(loc=0, scale=1.2))

    def decrease_weight(self):
        self.weight -= abs(np.random.normal(loc=0, scale=1.2))

    def disable_link(self):
        self.enabled = False