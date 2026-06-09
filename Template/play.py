"""
╔══════════════════════════════════════════════════════════════╗
║       PONG RL  ·  Mensch vs. KI-Agent (play.py)             ║
╚══════════════════════════════════════════════════════════════╝

AUSFÜHREN (nach dem Training):
  python play.py

STEUERUNG:
  ↑ / ↓       → Schläger hoch / runter
  Q / E       → Schläger kippen links / rechts  (wenn TILTING_ENABLED)
  R           → Runde neu starten
  ESC         → Beenden

DU (blau) spielst LINKS.
Der RL-AGENT (rot) spielt RECHTS.

WICHTIGER HINWEIS:
──────────────────
Der Agent wurde trainiert, auf der RECHTEN Seite zu spielen.
Die Beobachtung die er bekommt, muss EXAKT dem Format aus dem
Training entsprechen (pong_game.get_observation()).

Falls das Format abweicht → Agent verhält sich komisch!
Das ist ein häufiger Fehler beim Deployen von RL-Agenten ("distribution shift").
"""

import os
import sys
import pygame
import numpy as np
from stable_baselines3 import PPO
from config import *
from pong_game import PongGame
import sys
sys.stdout.reconfigure(encoding='utf-8')


# ── FARB-KONSTANTEN ───────────────────────────────────────────
C_BG         = (15,  15,  20)
C_WHITE      = (240, 240, 255)
C_GRAY       = (75,  75,  90)
C_DARK_GRAY  = (45,  45,  55)
C_PLAYER     = (80,  160, 255)   # Blau  – menschlicher Spieler
C_AGENT      = (255, 80,  80)    # Rot   – RL-Agent
C_YELLOW     = (255, 220, 60)
C_GREEN      = (80,  220, 120)

# ──────────────────────────────────────────────────────────────
#  BOT SPIELT GEGEN KI
# ──────────────────────────────────────────────────────────────

WATCH_BOT=True

# ──────────────────────────────────────────────────────────────
#  MODELL LADEN
# ──────────────────────────────────────────────────────────────

def load_model() -> PPO:
    """
    Lädt das trainierte PPO-Modell.

    Suchreihenfolge:
      1. Bestes Modell (nach Evaluierung)
      2. Finales Modell (Ende des Trainings)
      3. Neuester Checkpoint (falls Training unterbrochen)
    """
    candidates = [
        "./models/best_model.zip",
        MODEL_SAVE_PATH + ".zip",
    ]

    # Neueste Checkpoints hinzufügen
    if os.path.isdir(CHECKPOINT_PATH):
        ckpts = sorted(
            [f for f in os.listdir(CHECKPOINT_PATH) if f.endswith(".zip")],
            key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_PATH, f)),
            reverse=True,
        )
        candidates += [os.path.join(CHECKPOINT_PATH, f) for f in ckpts]

    for path in candidates:
        if os.path.exists(path):
            print(f"  ✓ Modell geladen: {path}")
            return PPO.load(path)

    # Kein Modell gefunden
    print("\n❌ Kein trainiertes Modell gefunden!")
    print("   Bitte zuerst ausführen:  python train.py\n")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
#  HILFSFUNKTIONEN FÜR DAS RENDERING
# ──────────────────────────────────────────────────────────────

def draw_paddle(screen, x: int, y: float, angle_deg: float, color: tuple):
    """Zeichnet einen (ggf. gekippten) Schläger."""
    surf = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, PADDLE_WIDTH, PADDLE_HEIGHT),
                     border_radius=4)
    if abs(angle_deg) > 0.5:
        surf = pygame.transform.rotate(surf, -angle_deg)
    rect = surf.get_rect(center=(int(x), int(y)))
    screen.blit(surf, rect)

def draw_wall(screen, x: int, y: float, angle_deg: float, color: tuple):
    """Zeichnet die Mittelwand."""
    surf = pygame.Surface((WALL_WIDTH, WALL_HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, WALL_WIDTH, WALL_HEIGHT),
                     border_radius=4)
    if abs(angle_deg) > 0.5:
        surf = pygame.transform.rotate(surf, -angle_deg)
    rect = surf.get_rect(center=(int(x), int(y)))
    screen.blit(surf, rect)

def draw_angle_arc(screen, x: int, y: float, angle_deg: float, color: tuple):
    """
    Zeigt den aktuellen Kippwinkel als kleinen Text an.
    Hilft dem Spieler zu sehen, wie weit der Schläger gekippt ist.
    """
    font  = pygame.font.Font(None, 20)
    sign  = "↺" if angle_deg < 0 else ("↻" if angle_deg > 0 else "│")
    label = f"{sign} {abs(angle_deg):.0f}°"
    surf  = font.render(label, True, (*color, 180))
    screen.blit(surf, (x - 22, y + PADDLE_HEIGHT // 2 + 6))


def draw_dashed_line(screen, color, x, y_start, y_end, dash=12, gap=10):
    """Zeichnet eine gestrichelte vertikale Linie."""
    y = y_start
    while y < y_end:
        pygame.draw.rect(screen, color, (x - 2, y, 4, dash))
        y += dash + gap


def draw_result_overlay(screen, fonts, winner: str, score_p: int, score_a: int):
    """Zeigt ein semi-transparentes Ergebnis-Overlay an."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 155))
    screen.blit(overlay, (0, 0))

    color = C_PLAYER if winner == "player" else C_AGENT
    text  = "🎉 DU GEWINNST!" if winner == "player" else "🤖 KI GEWINNT!"
    lbl   = fonts["big"].render(text, True, color)
    screen.blit(lbl, (WINDOW_WIDTH // 2 - lbl.get_width() // 2,
                       WINDOW_HEIGHT // 2 - 50))

    sub = fonts["medium"].render(
        f"Gesamt  Du {score_p} : {score_a} KI",
        True, C_WHITE
    )
    screen.blit(sub, (WINDOW_WIDTH // 2 - sub.get_width() // 2,
                       WINDOW_HEIGHT // 2 + 20))

    hint = fonts["small"].render("Weiter in Kürze  |  R = sofort neu starten",
                                  True, C_GRAY)
    screen.blit(hint, (WINDOW_WIDTH // 2 - hint.get_width() // 2,
                        WINDOW_HEIGHT // 2 + 65))


# ──────────────────────────────────────────────────────────────
#  HAUPT-SPIELSCHLEIFE
# ──────────────────────────────────────────────────────────────

def play():
    print("\n" + "═" * 55)
    print("  PONG: MENSCH (←) vs. RL-AGENT (→)")
    print("═" * 55)
    print("  Steuerung:")
    print("    w / s       →  Schläger bewegen")
    if TILTING_ENABLED:
        print("    Q / E       →  Schläger kippen")
    print("    R           →  Runde neu starten")
    print("    ESC         →  Beenden")
    print("═" * 55)

    # Modell laden
    model = load_model()

    # Pygame initialisieren
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pong – Du vs. RL-Agent")
    clock  = pygame.time.Clock()

    fonts = {
        "big":    pygame.font.Font(None, 90),
        "score":  pygame.font.Font(None, 82),
        "medium": pygame.font.Font(None, 38),
        "small":  pygame.font.Font(None, 24),
        "tiny":   pygame.font.Font(None, 20),
    }

    # Spielzustand
    game         = PongGame(tilting_enabled=TILTING_ENABLED)
    score_player = 0     # Gesamtsiege des Menschen
    score_agent  = 0     # Gesamtsiege der KI
    last_winner  = None  # "player" oder "agent"
    show_result  = False
    result_timer = 0     # Millisekunden bis zum Auto-Neustart
    running      = True
    c_reward     = 0

    game.reset()

    print("\n  Spiel startet! Viel Spaß!\n")

    while running:
        dt = clock.tick(FPS)   # Millisekunden seit letztem Frame

        # ── EVENTS ──────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    show_result = False

        # ── SPIELLOGIK (nur wenn kein Ergebnis-Screen) ───────
        if not show_result:
            # 1. Menschliche Eingabe lesen
            if WATCH_BOT:
                game.apply_bot_action()
            else:
                keys = pygame.key.get_pressed()

                # Bewegung und Kippen getrennt auswerten
                move_action = PongGame.ACTION_NOTHING
                tilt_action = PongGame.ACTION_NOTHING

                if keys[pygame.K_w]:
                    move_action = PongGame.ACTION_UP
                elif keys[pygame.K_s]:
                    move_action = PongGame.ACTION_DOWN

                if keys[pygame.K_q] and TILTING_ENABLED:
                    tilt_action = PongGame.ACTION_TILT_CCW
                elif keys[pygame.K_e] and TILTING_ENABLED:
                    tilt_action = PongGame.ACTION_TILT_CW

                # Beide unabhängig anwenden
                game.apply_player_action(move_action)
                game.apply_player_action(tilt_action)



            # 2. Agent-Aktion berechnen
            #    WICHTIG: get_observation() muss exakt das gleiche Format
            #    liefern wie während des Trainings in pong_env.py!
            obs = game.get_observation()
            agent_action, _ = model.predict(obs, deterministic=True)
            # deterministic=True → Agent wählt immer die Aktion mit der
            # höchsten Wahrscheinlichkeit (keine Exploration mehr)
            game.apply_agent_action(int(agent_action))

            # 3. Physik-Update
            reward = game.step_physics()
            c_reward+=reward

            # 4. Runde beendet?
            if game.is_done(reward):
                if reward > 0:
                    score_agent  += 1
                    last_winner   = "agent"
                else:
                    score_player += 1
                    last_winner   = "player"

                show_result  = True
                result_timer = 2500   # 2.5 Sekunden anzeigen

                print(f"  {'KI' if last_winner == 'agent' else 'Du'} gewinnt!  "
                      f"Gesamt: Du {score_player} – {score_agent} KI")

        else:
            # Countdown für Auto-Neustart
            result_timer -= dt
            if result_timer <= 0:
                game.reset()
                c_reward=0
                show_result = False

        # ── RENDERING ────────────────────────────────────────
        screen.fill(C_BG)

        # Mittellinie
        draw_dashed_line(screen, C_DARK_GRAY,
                         WINDOW_WIDTH // 2, 0, WINDOW_HEIGHT)

        # Ball
        pygame.draw.circle(
            screen, C_WHITE,
            (int(game.ball_x), int(game.ball_y)),
            BALL_SIZE // 2
        )

        # Schläger
        draw_paddle(screen, PLAYER_PADDLE_X, game.player_y,
                    game.player_angle, C_PLAYER)
        draw_paddle(screen, AGENT_PADDLE_X,  game.agent_y,
                    game.agent_angle,  C_AGENT)
        draw_wall(screen, game.wall_x,game.wall_y,0, C_WHITE)

        # Kippwinkel-Anzeige
        if TILTING_ENABLED:
            draw_angle_arc(screen, PLAYER_PADDLE_X - 5, game.player_y,
                           game.player_angle, C_PLAYER)
            draw_angle_arc(screen, AGENT_PADDLE_X - 5,  game.agent_y,
                           game.agent_angle,  C_AGENT)

        # Spielstand (Siege, nicht Punkte pro Runde)
        p_score = fonts["score"].render(str(score_player), True, C_PLAYER)
        a_score = fonts["score"].render(str(score_agent),  True, C_AGENT)
        screen.blit(p_score, (WINDOW_WIDTH // 4  - p_score.get_width() // 2, 12))
        screen.blit(a_score, (3 * WINDOW_WIDTH // 4 - a_score.get_width() // 2, 12))

        # Labels
        if not WATCH_BOT:
            you_lbl = fonts["small"].render("DU", True, C_PLAYER)
        if WATCH_BOT:
            you_lbl = fonts["small"].render("BOT", True, C_PLAYER)

        ki_lbl  = fonts["small"].render("KI", True, C_AGENT)
        screen.blit(you_lbl, (WINDOW_WIDTH // 4  - you_lbl.get_width() // 2, 96))
        screen.blit(ki_lbl,  (3 * WINDOW_WIDTH // 4 - ki_lbl.get_width()  // 2, 96))

        reward_txt = fonts["tiny"].render(f"r = {c_reward:+.2f}", True, C_AGENT)
        screen.blit(reward_txt, (3 * WINDOW_WIDTH // 4 - reward_txt.get_width() // 2, 118))

        # Steuerungshinweis (unten)
        ctrl = "↑↓ mit w/s: Bewegen"
        if TILTING_ENABLED:
            ctrl += "  Q/E: Kippen"
        ctrl += "  R: Neustart  ESC: Beenden"
        hint = fonts["small"].render(ctrl, True, C_GRAY)
        screen.blit(hint, (WINDOW_WIDTH // 2 - hint.get_width() // 2,
                            WINDOW_HEIGHT - 24))

        # Ergebnis-Overlay (wenn Runde beendet)
        if show_result and last_winner:
            draw_result_overlay(screen, fonts, last_winner,
                                score_player, score_agent)

        pygame.display.flip()

    # ── AUFRÄUMEN ────────────────────────────────────────────
    pygame.quit()
    print(f"\n  Endstand → Du: {score_player}  |  KI: {score_agent}")
    print("  Auf Wiedersehen!\n")


# ──────────────────────────────────────────────────────────────
#  EINSTIEGSPUNKT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    play()
