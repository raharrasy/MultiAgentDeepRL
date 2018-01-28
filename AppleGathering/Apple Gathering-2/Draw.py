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

        # draw opponent beams
        for (x,y) in self.state.beamed_list:
            rect = pygame.Rect(x+(self.sight_radius)*2,y+(self.sight_radius)*2,2,2)
            pygame.draw.rect(self.screen,(255,0,255),rect)

        # draw player locations

        alive_players = [player for player in self.state.player_list]
        for player in alive_players:
            rect = pygame.Rect(player.x+(self.sight_radius)*2,player.y+(self.sight_radius)*2,2,2)
            pygame.draw.rect(self.screen,player.color,rect)

        # draw food

        for food in self.state.food_list:
            rect = pygame.Rect(food.x+(self.sight_radius)*2,food.y+(self.sight_radius)*2,2,2)
            pygame.draw.rect(self.screen,food.color,rect)
