
import pygame
import sys
import random

WIDTH, HEIGHT = 900, 600
PADDLE_W, PADDLE_H = 14, 100
BALL_SIZE = 14
PADDLE_SPEED = 8
BALL_SPEED = 7
FONT_NAME = "arial"

WHITE = (240, 240, 240)
BG = (25, 25, 30)
DIM = (120, 120, 140)
ACCENT = (120, 200, 255)

class Paddle(pygame.Rect):
    def __init__(self, x, y):
        super().__init__(x, y, PADDLE_W, PADDLE_H)
        self.speed = 0

    def move(self, dy):
        self.y += dy
        self.y = max(0, min(self.y, HEIGHT - PADDLE_H))

class Ball(pygame.Rect):
    def __init__(self, x, y):
        super().__init__(x, y, BALL_SIZE, BALL_SIZE)
        self.vx = random.choice([-1, 1]) * BALL_SPEED
        self.vy = random.choice([-1, 1]) * (BALL_SPEED - 2)

    def reset(self):
        self.center = (WIDTH // 2, HEIGHT // 2)
        self.vx = random.choice([-1, 1]) * BALL_SPEED
        self.vy = random.choice([-1, 1]) * (BALL_SPEED - 2)

def draw_center_dashed_line(surface):
    dash_h = 18
    gap = 12
    x = WIDTH // 2 - 2
    for y in range(0, HEIGHT, dash_h + gap):
        pygame.draw.rect(surface, (70, 70, 80), (x, y, 4, dash_h), border_radius=2)

def reflect_ball_from_paddle(ball, paddle):
    # Compute relative hit position and add a bit of "spin"
    rel = (ball.centery - paddle.centery) / (PADDLE_H / 2)
    rel = max(-1, min(1, rel))
    speed = (abs(ball.vx) + abs(ball.vy)) * 0.5
    speed = max(BALL_SPEED, min(speed + 0.5, 12))
    ball.vx = -ball.vx
    ball.vy = rel * speed

def cpu_policy(ball, paddle):
    # Simple predictive tracking
    target_y = ball.centery
    if abs(target_y - paddle.centery) > 6:
        return PADDLE_SPEED if target_y > paddle.centery else -PADDLE_SPEED
    return 0

def game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont(FONT_NAME, 20)
    font_big = pygame.font.SysFont(FONT_NAME, 54, bold=True)

    left = Paddle(30, HEIGHT//2 - PADDLE_H//2)
    right = Paddle(WIDTH - 30 - PADDLE_W, HEIGHT//2 - PADDLE_H//2)
    ball = Ball(WIDTH//2 - BALL_SIZE//2, HEIGHT//2 - BALL_SIZE//2)
    score_l, score_r = 0, 0

    cpu_enabled = True
    paused = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_c:
                    cpu_enabled = not cpu_enabled

        keys = pygame.key.get_pressed()
        # Left paddle controls
        dy = 0
        if keys[pygame.K_w]: dy -= PADDLE_SPEED
        if keys[pygame.K_s]: dy += PADDLE_SPEED
        left.move(dy)

        # Right paddle: CPU or manual
        if cpu_enabled:
            right.move(cpu_policy(ball, right))
        else:
            dy_r = 0
            if keys[pygame.K_UP]: dy_r -= PADDLE_SPEED
            if keys[pygame.K_DOWN]: dy_r += PADDLE_SPEED
            right.move(dy_r)

        if not paused:
            # Move ball
            ball.x += int(ball.vx)
            ball.y += int(ball.vy)

            # Collide with top/bottom
            if ball.top <= 0 or ball.bottom >= HEIGHT:
                ball.vy = -ball.vy

            # Collide with paddles
            if ball.colliderect(left) and ball.vx < 0:
                reflect_ball_from_paddle(ball, left)
                ball.left = left.right
            elif ball.colliderect(right) and ball.vx > 0:
                reflect_ball_from_paddle(ball, right)
                ball.right = right.left

            # Score
            if ball.right < 0:
                score_r += 1
                ball.reset()
            elif ball.left > WIDTH:
                score_l += 1
                ball.reset()

        # Draw
        screen.fill(BG)
        draw_center_dashed_line(screen)
        pygame.draw.rect(screen, WHITE, left, border_radius=4)
        pygame.draw.rect(screen, WHITE, right, border_radius=4)
        pygame.draw.ellipse(screen, ACCENT, ball)

        # UI
        score_text = font_big.render(f"{score_l}   {score_r}", True, WHITE)
        screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))
        info = "P: pause | C: CPU on" if cpu_enabled else "P: pause | C: CPU off"
        info_text = font_small.render(info, True, DIM)
        screen.blit(info_text, (20, HEIGHT - 28))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    game()
