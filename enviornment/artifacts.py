class Artifact:

    def __init__(self, pos=None):
        self.pos = pos

    @staticmethod
    def kind():
        pass

    @classmethod
    def image_path(cls):
        return f'img/{cls.kind()}.png'

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos
    

class Agent(Artifact):
    @staticmethod
    def kind():
        return 'a'
    
class Robot(Artifact):
    @staticmethod
    def kind():
        return 'r'
    
    def get_score(self):
        return self.score

class Goal(Artifact):
    @staticmethod
    def kind():
        return 'x'
    
class TennisBallA(Artifact):
    @staticmethod
    def kind():
        return '1'
    
class TennisBallB(Artifact):
    @staticmethod
    def kind():
        return '2'
    
