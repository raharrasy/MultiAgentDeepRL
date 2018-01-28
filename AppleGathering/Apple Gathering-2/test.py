import pygame
pygame.init()
key = pygame.key.get_pressed()
while True:
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == pygame.K_w:
            print('Forward')
        elif event.type == KEYDOWN and event.key == pygame.K_s:
            print('Backward')  