import pygame
import random

class DinoGame:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 600, 200
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game elements
        self.dino = pygame.Rect(50, 130, 30, 60)
        self.ground = 190
        self.obstacles = []
        self.score = 0
        self.jump = False
        self.jump_speed = 15
        self.running = True  # Game loop control

    def add_obstacle(self):
        if random.random() < 0.02 and len(self.obstacles) < 1:
            self.obstacles.append(pygame.Rect(self.WIDTH, 140, 20, 20))

    def update(self, action):
        # Action: 0 = no jump, 1 = jump
        if action == 1 and not self.jump and self.dino.y == 130:
            self.jump = True

        # Game physics
        if self.jump:
            self.dino.y -= self.jump_speed
            self.jump_speed -= 1
            if self.jump_speed < -15:
                self.jump = False
                self.jump_speed = 15
                self.dino.y = 130

        # Update obstacles
        for obs in self.obstacles:
            obs.x -= 5
            if obs.colliderect(self.dino):
                return -1  # Game over
            if obs.x < -20:
                self.obstacles.remove(obs)
                self.score += 1

        self.add_obstacle()
        return 1  # Continue playing

    def get_state(self):
        # State: [dino_y, obstacle_distance, obstacle_height]
        if self.obstacles:
            return [self.dino.y, self.obstacles[0].x - self.dino.x, 20]
        return [self.dino.y, 999, 0]  # No obstacles

    def render(self):
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 0), self.dino)
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), obs)
        pygame.display.flip()
        self.clock.tick(30)

        # Event handling to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                exit()
