
import random
import pygame
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. TRAIN THE ARTIFICIAL NEURAL NETWORK
# -------------------------------

def create_training_data():
    X = []
    y = []

    # Generate example situations for the ANN to learn from.
    for _ in range(1500):
        player_health = random.randint(1, 100)
        enemy_health = random.randint(1, 100)
        distance = random.randint(0, 500)

        # 1 means the player is attacking the enemy.
        enemy_attacking = random.randint(0, 1)

        if enemy_attacking == 1:
            action = "Defend"
        elif distance > 120:
            action = "Move"
        elif enemy_health < 25 and player_health > 40:
            action = "Defend"
        else:
            action = "Attack"

        X.append([
            player_health,
            enemy_health,
            distance,
            enemy_attacking
        ])
        y.append(action)

    return np.array(X), np.array(y)


X_train, y_train = create_training_data()

ann_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ann", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        max_iter=500,
        random_state=42
    ))
])

ann_model.fit(X_train, y_train)

print("ANN training completed successfully!")


# -------------------------------
# 2. INITIALIZE PYGAME
# -------------------------------

pygame.init()

WIDTH = 900
HEIGHT = 550
GROUND_Y = 430

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stickman Fighting Game - ANN AI")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 22)
small_font = pygame.font.SysFont("Arial", 17)

WHITE = (255, 255, 255)
BLACK = (20, 20, 25)
BLUE = (50, 130, 255)
RED = (235, 65, 65)
GREEN = (50, 200, 90)
DARK_GREEN = (30, 110, 50)
GREY = (110, 110, 120)
YELLOW = (255, 210, 60)


# -------------------------------
# 3. GAME VARIABLES
# -------------------------------

def reset_game():
    return {
        "player_x": 170,
        "enemy_x": 700,
        "player_health": 100,
        "enemy_health": 100,
        "player_attack_timer": 0,
        "enemy_attack_timer": 0,
        "player_cooldown": 0,
        "enemy_cooldown": 0,
        "enemy_action": "Move",
        "game_over": False,
        "winner": ""
    }


game = reset_game()


# -------------------------------
# 4. DRAW A STICKMAN
# -------------------------------

def draw_stickman(x, color, attacking=False, defending=False):
    head_y = GROUND_Y - 90
    body_top = GROUND_Y - 68
    body_bottom = GROUND_Y - 30

    # Head
    pygame.draw.circle(
        screen, color, (int(x), head_y), 12, 3
    )

    # Body
    pygame.draw.line(
        screen, color,
        (int(x), body_top),
        (int(x), body_bottom), 4
    )

    # Arms
    if attacking:
        pygame.draw.line(
            screen, color,
            (int(x), body_top + 8),
            (int(x + 35), body_top + 15), 4
        )
        pygame.draw.line(
            screen, color,
            (int(x), body_top + 8),
            (int(x - 15), body_top + 25), 4
        )
    elif defending:
        pygame.draw.line(
            screen, color,
            (int(x), body_top + 8),
            (int(x - 18), body_top + 2), 4
        )
        pygame.draw.line(
            screen, color,
            (int(x), body_top + 8),
            (int(x + 18), body_top + 2), 4
        )
    else:
        pygame.draw.line(
            screen, color,
            (int(x), body_top + 8),
            (int(x - 20), body_top + 25), 4
        )
        pygame.draw.line(
            screen, color,
            (int(x), body_top + 8),
            (int(x + 20), body_top + 25), 4
        )

    # Legs
    pygame.draw.line(
        screen, color,
        (int(x), body_bottom),
        (int(x - 16), GROUND_Y), 4
    )
    pygame.draw.line(
        screen, color,
        (int(x), body_bottom),
        (int(x + 16), GROUND_Y), 4
    )


# -------------------------------
# 5. DRAW HEALTH BARS
# -------------------------------

def draw_health_bar(x, y, health, label):
    pygame.draw.rect(
        screen, GREY, (x, y, 250, 22)
    )

    health_width = int(250 * max(0, health) / 100)

    pygame.draw.rect(
        screen, GREEN, (x, y, health_width, 22)
    )

    pygame.draw.rect(
        screen, WHITE, (x, y, 250, 22), 2
    )

    text = font.render(
        f"{label}: {max(0, health)} HP", True, WHITE
    )
    screen.blit(text, (x, y - 29))


# -------------------------------
# 6. MAIN GAME LOOP
# -------------------------------

running = True

while running:
    dt = clock.tick(60) / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_r:
                game = reset_game()

            if (
                event.key == pygame.K_f
                and not game["game_over"]
                and game["player_cooldown"] <= 0
            ):
                game["player_attack_timer"] = 0.22
                game["player_cooldown"] = 0.55

                distance = abs(
                    game["player_x"] - game["enemy_x"]
                )

                if distance < 100:
                    # A defending enemy blocks the attack.
                    if game["enemy_action"] != "Defend":
                        game["enemy_health"] -= 12

    keys = pygame.key.get_pressed()

    if not game["game_over"]:
        # Reduce attack timers and cooldowns.
        game["player_attack_timer"] = max(
            0, game["player_attack_timer"] - dt
        )
        game["enemy_attack_timer"] = max(
            0, game["enemy_attack_timer"] - dt
        )
        game["player_cooldown"] = max(
            0, game["player_cooldown"] - dt
        )
        game["enemy_cooldown"] = max(
            0, game["enemy_cooldown"] - dt
        )

        # Player movement.
        if keys[pygame.K_a]:
            game["player_x"] -= 250 * dt

        if keys[pygame.K_d]:
            game["player_x"] += 250 * dt

        game["player_x"] = max(
            35, min(WIDTH - 35, game["player_x"])
        )

        # Keep the two characters from overlapping.
        distance = abs(
            game["player_x"] - game["enemy_x"]
        )

        # ANN chooses the enemy's next action.
        # The last feature represents an incoming player attack.
        incoming_attack = int(
            game["player_attack_timer"] > 0
        )

        features = np.array([[
            game["player_health"],
            game["enemy_health"],
            distance,
            incoming_attack
        ]])

        game["enemy_action"] = ann_model.predict(
            features
        )[0]

        # Enemy movement.
        if game["enemy_action"] == "Move" and distance > 90:
            if game["enemy_x"] > game["player_x"]:
                game["enemy_x"] -= 140 * dt
            else:
                game["enemy_x"] += 140 * dt

        # Enemy attack.
        elif (
            game["enemy_action"] == "Attack"
            and distance < 105
            and game["enemy_cooldown"] <= 0
        ):
            game["enemy_attack_timer"] = 0.25
            game["enemy_cooldown"] = 0.9

            # A defending player takes less damage.
            if game["player_attack_timer"] <= 0:
                game["player_health"] -= 8

        # Finish the game when either health bar reaches zero.
        if game["player_health"] <= 0:
            game["player_health"] = 0
            game["game_over"] = True
            game["winner"] = "AI ENEMY WINS!"

        elif game["enemy_health"] <= 0:
            game["enemy_health"] = 0
            game["game_over"] = True
            game["winner"] = "PLAYER WINS!"

    # -------------------------------
    # 7. DRAW THE GAME SCREEN
    # -------------------------------

    screen.fill(BLACK)

    # Arena floor.
    pygame.draw.line(
        screen, GREY,
        (0, GROUND_Y + 2),
        (WIDTH, GROUND_Y + 2), 4
    )

    # Health bars.
    draw_health_bar(
        35, 55, game["player_health"], "PLAYER"
    )
    draw_health_bar(
        WIDTH - 285, 55, game["enemy_health"], "AI ENEMY"
    )

    # Character labels.
    screen.blit(
        small_font.render("BLUE: PLAYER", True, BLUE),
        (game["player_x"] - 55, GROUND_Y + 15)
    )
    screen.blit(
        small_font.render("RED: ANN ENEMY", True, RED),
        (game["enemy_x"] - 65, GROUND_Y + 15)
    )

    # Draw both stickmen.
    draw_stickman(
        game["player_x"],
        BLUE,
        attacking=game["player_attack_timer"] > 0
    )

    draw_stickman(
        game["enemy_x"],
        RED,
        attacking=game["enemy_attack_timer"] > 0,
        defending=game["enemy_action"] == "Defend"
    )

    # Show the ANN decision.
    action_text = font.render(
        f"ANN Action: {game['enemy_action']}",
        True, YELLOW
    )
    screen.blit(action_text, (350, 120))

    # Instructions.
    controls = small_font.render(
        "A/D: Move   F: Attack   R: Restart   ESC: Quit",
        True, WHITE
    )
    screen.blit(controls, (220, HEIGHT - 35))

    # Game over message.
    if game["game_over"]:
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))

        message = font.render(
            game["winner"], True, YELLOW
        )
        restart_message = font.render(
            "Press R to play again", True, WHITE
        )

        screen.blit(
            message,
            message.get_rect(
                center=(WIDTH // 2, HEIGHT // 2 - 20)
            )
        )
        screen.blit(
            restart_message,
            restart_message.get_rect(
                center=(WIDTH // 2, HEIGHT // 2 + 25)
            )
        )

    pygame.display.flip()

pygame.quit()
print("Game closed.")
