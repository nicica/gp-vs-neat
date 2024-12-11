import inspect
import random
import sys
from enum import Enum

import pygame.surfarray

import enviornment.config as config
from enviornment.artifacts import Agent, Goal, Robot, TennisBallA, TennisBallB
from enviornment.image import Image
from enviornment.tiles import TilesFactory, Hole, Grass


class Quit(Exception):
    pass


class Action(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


class Environment:
    def __init__(self, path):
        self.path = path
        self.start_position = {}
        self.field_map = []
        self.artifacts_map = {obj.kind(): obj()
                              for name, obj in sorted(inspect.getmembers(sys.modules['enviornment.artifacts']), reverse=True)
                              if inspect.isclass(obj) and name != 'Artifact'}
        with open(path, 'r') as file:
            for i, line in enumerate(file):
                self.field_map.append([])
                for j, c in enumerate(line.strip()):
                    self.field_map[-1].append(TilesFactory.generate_tile(c
                                                                         if c not in self.artifacts_map.keys()
                                                                         else Grass.kind(), [i, j]))
                    if c in self.artifacts_map.keys():
                        self.artifacts_map[c].set_position([i, j])
        self.all_actions = [act for act in Action]
        for key in self.artifacts_map:
            if self.artifacts_map[key].get_position() is None and key in {Agent.kind(), Goal.kind(), Robot.kind()}:
                raise Exception(f'Environment map is missing agent, robot or goal!')
            if key == Agent.kind() or Robot.kind() or TennisBallA.kind() or TennisBallB.kind():
                self.start_position[key] = self.artifacts_map[key].get_position().copy()

        self.display = None
        self.clock = None

    def __del__(self):
        if self.display:
            pygame.quit()

    def full_reset(self):
        self.__init__(self.path)

    def reset(self, kind):
        self.artifacts_map[kind].set_position(self.start_position[kind].copy())

    def get_agent_position(self):
        return self.artifacts_map[Agent.kind()].get_position()
    
    def get_robot_position(self):
        return self.artifacts_map[Robot.kind()].get_position()

    def get_goal_position(self):
        return self.artifacts_map[Goal.kind()].get_position()

    def get_artifact_position(self, kind):
        return self.artifacts_map[kind].get_position()

    def get_all_actions(self):
        return self.all_actions

    def get_euklid_distance_tennis_ball_a(self):
        dif_1 = abs(self.artifacts_map[Robot.kind()].get_position()[0] - self.artifacts_map[TennisBallA.kind()].get_position()[0])
        dif_2 = abs(self.artifacts_map[Robot.kind()].get_position()[1] - self.artifacts_map[TennisBallA.kind()].get_position()[1])
        return dif_1 + dif_2
    
    def get_euklid_distance_tennis_ball_b(self):
        dif_b_1 = abs(self.artifacts_map[Robot.kind()].get_position()[0] - self.artifacts_map[TennisBallB.kind()].get_position()[0])
        dif_b_2 = abs(self.artifacts_map[Robot.kind()].get_position()[1] - self.artifacts_map[TennisBallB.kind()].get_position()[1])
        return dif_b_1 + dif_b_2
    
    def get_euklid_distance_house(self):
        dif_1 = abs(self.artifacts_map[Robot.kind()].get_position()[0] - self.artifacts_map[Goal.kind()].get_position()[0])
        dif_2 = abs(self.artifacts_map[Robot.kind()].get_position()[1] - self.artifacts_map[Goal.kind()].get_position()[1])
        return dif_1 + dif_2

    def get_random_action(self):
        return self.all_actions[random.randint(0, len(self.all_actions) - 1)]

    def get_field_map(self):
        return self.field_map

    def render_textual(self):
        """
            Rendering textual representation of the current state of the environment.
        """
        text = ''.join([''.join([t.kind() for t in row]) for row in self.field_map])
        gp = (len(self.field_map[0]) * self.artifacts_map[Goal.kind()].get_position()[0] +
              self.artifacts_map[Goal.kind()].get_position()[1])
        text = text[:gp] + Goal.kind() + text[gp + 1:]
        ap = (len(self.field_map[0]) * self.artifacts_map[Agent.kind()].get_position()[0] +
              self.artifacts_map[Agent.kind()].get_position()[1])
        text = text[:ap] + Agent.kind() + text[ap + 1:]
        cols = len(self.field_map[0])
        print('\n'.join(text[i:i + cols] for i in range(0, len(text), cols)))

    def render(self, fps):
        """
            Rendering current state of the environment in FPS (frames per second).
        """
        if not self.display:
            pygame.init()
            pygame.display.set_caption('Pyppy Adventure')
            self.display = pygame.display.set_mode((len(self.field_map[0]) * config.TILE_SIZE,
                                                    len(self.field_map) * config.TILE_SIZE))
            self.clock = pygame.time.Clock()
        for i in range(len(self.field_map)):
            for j in range(len(self.field_map[0])):
                self.display.blit(Image.get_image(self.field_map[i][j].image_path(),
                                                  (config.TILE_SIZE, config.TILE_SIZE)),
                                  (j * config.TILE_SIZE, i * config.TILE_SIZE))
        for a in self.artifacts_map:
            self.display.blit(Image.get_image(self.artifacts_map[a].image_path(),
                                              (config.TILE_SIZE, config.TILE_SIZE), config.GREEN),
                              (self.artifacts_map[self.artifacts_map[a].kind()].get_position()[1] * config.TILE_SIZE,
                               self.artifacts_map[self.artifacts_map[a].kind()].get_position()[0] * config.TILE_SIZE))
        pygame.display.flip()
        self.clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise Quit

    def step(self, action, kind, score):
        """
            Actions: UP(0), LEFT(1), DOWN(2), RIGHT(3)
            Returns: new_state, reward, done
                new_state - agent position in new state obtained by applying chosen action in the current state.
                reward - immediate reward awarded for transitioning to the new state.
                done - boolean flag specifying whether the new state is terminal state.
        """
        if action not in self.all_actions:
            raise Exception(f'Illegal action {action}! Legal actions: {self.all_actions}.')
        if action == Action.UP and self.artifacts_map[kind].get_position()[0] > 0:
            self.artifacts_map[kind].get_position()[0] -= 1
        elif action == Action.LEFT and self.artifacts_map[kind].get_position()[1] > 0:
            self.artifacts_map[kind].get_position()[1] -= 1
        elif action == Action.DOWN and self.artifacts_map[kind].get_position()[0] < len(self.field_map) - 1:
            self.artifacts_map[kind].get_position()[0] += 1
        elif action == Action.RIGHT and self.artifacts_map[kind].get_position()[1] < len(self.field_map[0]) - 1:
            self.artifacts_map[kind].get_position()[1] += 1

        agent_position = self.artifacts_map[kind].get_position()
        reward = self.field_map[agent_position[0]][agent_position[1]].reward()

        new_score = score


        done = (agent_position == self.artifacts_map[Goal.kind()].get_position() and score>=5)

        if agent_position == self.artifacts_map[TennisBallA.kind()].get_position():
            new_score += 1
            while True:
                random_position = [random.randint(0,6),random.randint(0, 11)]
                if random_position==agent_position or random_position==self.artifacts_map[Goal.kind()].get_position() or self.field_map[random_position[0]][random_position[1]].kind() != Grass.kind():
                    continue
                self.artifacts_map[TennisBallA.kind()].set_position( random_position)
                break
        elif agent_position == self.artifacts_map[TennisBallB.kind()].get_position():
            new_score += 1
            while True:
                random_position = [random.randint(0,6),random.randint(0, 11)]
                if random_position==agent_position or random_position==self.artifacts_map[Goal.kind()].get_position() or self.field_map[random_position[0]][random_position[1]].kind() != Grass.kind():
                    continue
                self.artifacts_map[TennisBallB.kind()].set_position( random_position)
                break

        
        return agent_position, reward, done, new_score
