from gameplay import State
import pygame

class Display(object) :
    def __init__(self, screen, state, sight_radius):
        self.screen = screen
        self.state = state
        self.sight_radius = sight_radius

    def setState(self,state):
        self.state = state

    def drawState(self):
        # draw margins
        x_size = self.state.x_size
        y_size = self.state.y_size
        margins1 = [((i+self.sight_radius)*2,2*(self.sight_radius)) for i in range(0,x_size)]
        margins2 = [((i+self.sight_radius)*2,2*(y_size+self.sight_radius-1)) for i in range(0,x_size)]
        margins3 = [(2*(self.sight_radius),(j+self.sight_radius)*2) for j in range(0,y_size)]
        margins4 = [(2*(self.sight_radius+x_size-1),(j+self.sight_radius)*2) for j in range(0,y_size)]
        
        for (x,y) in margins1:
            rect = pygame.Rect(x,y,2,2)
            pygame.draw.rect(self.screen,(0,255,255),rect)

        for (x,y) in margins2:
            rect = pygame.Rect(x,y,2,2)
            pygame.draw.rect(self.screen,(0,255,255),rect)

        for (x,y) in margins3:
            rect = pygame.Rect(x,y,2,2)
            pygame.draw.rect(self.screen,(0,255,255),rect)

        for (x,y) in margins4:
            rect = pygame.Rect(x,y,2,2)
            pygame.draw.rect(self.screen,(0,255,255),rect)

        # draw the obstacles prescribed by the map
        for (x,y) in self.state.obstacleCoord:
            rect = pygame.Rect(2*(self.sight_radius)+x,2*(self.sight_radius)+y,2,2)
            pygame.draw.rect(self.screen,(0,255,255),rect)

        # draw player locations

        alive_players = [player for player in self.state.player_list]
        for player in alive_players:
            rect = pygame.Rect(2*(self.sight_radius)+player.x,2*(self.sight_radius)+player.y,2,2)
            pygame.draw.rect(self.screen,player.color,rect)

        # draw food

        for food in self.state.food_list:
            if not food.is_dead:
                rect = pygame.Rect(2*(self.sight_radius)+food.x,2*(self.sight_radius)+food.y,2,2)
                pygame.draw.rect(self.screen,food.color,rect)
