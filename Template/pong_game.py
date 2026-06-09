"""
╔══════════════════════════════════════════════════════════════╗
║          PONG RL  ·  Spielphysik (pong_game.py)             ║
╚══════════════════════════════════════════════════════════════╝

Diese Datei enthält die reine Spielphysik von Pong.
Sie wird von zwei anderen Dateien verwendet:
  • pong_env.py  → Gymnasium-Umgebung für das RL-Training
  • play.py      → Menschliches Spiel gegen den Agenten

Durch diese Trennung gibt es keine Code-Duplikation und die
Physik ist immer konsistent.
"""

import math
import numpy as np
from config import *

import time


class PongGame:
    """
    Die reine Spiellogik von Pong (ohne Pygame, ohne RL).

    Enthält:
      - Ballbewegung und Wandkollisionen
      - Schlägerbewegung mit optionalem Kippen
      - Physikalisch korrekte Reflexion an gekippten Schlägern
      - Normalisierte Beobachtungen für das neuronale Netz
    """

    # Aktions-Konstanten (leserlicher als Zahlen 0-4)
    ACTION_NOTHING   = 0
    ACTION_UP        = 1
    ACTION_DOWN      = 2
    ACTION_TILT_CCW  = 3   # Gegen Uhrzeigersinn (counterclockwise)
    ACTION_TILT_CW   = 4   # Im Uhrzeigersinn (clockwise)

    def __init__(self, tilting_enabled: bool = TILTING_ENABLED):
        self.tilting_enabled = tilting_enabled
        self.reset()

    # ──────────────────────────────────────────────────────────
    #  ZUSTAND
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Setzt das Spiel komplett zurück (neue Episode)."""

        # Zufällige Startrichtung für den Ball
        # → Agent soll nicht nur für eine Richtung lernen
        angle_rad = np.random.uniform(-20, 20) * math.pi / 180
        direction = float(np.random.choice([-1.0, 1.0]))   # links oder rechts

        # Ball
        self.ball_x  = float(WINDOW_WIDTH  / 2)
        self.ball_y  = float(WINDOW_HEIGHT / 2)
        self.ball_vx = direction * BALL_INITIAL_SPEED * math.cos(angle_rad)
        self.ball_vy = BALL_INITIAL_SPEED * math.sin(angle_rad)

        # Schläger: y-Koordinate der Mitte
        self.player_y = float(WINDOW_HEIGHT / 2)   # Links  (Mensch / Bot)
        self.agent_y  = float(WINDOW_HEIGHT / 2)   # Rechts (RL-Agent)

        self.wall_y= float(WINDOW_HEIGHT / 3)   # Y-Position der Wand
        self.wall_x=float(WINDOW_WIDTH / 2)   # X-Position der Wand
        self.wall_moving_direction=np.random.choice([1,-1])

        # Kippwinkel in Grad (0 = senkrecht)
        self.player_angle = 0.0
        self.agent_angle  = 0.0
        self.last_hit=""
        self.new_hit=True

        # Zähler
        self.steps = 0

    # ──────────────────────────────────────────────────────────
    #  AKTIONEN ANWENDEN
    # ──────────────────────────────────────────────────────────

    def apply_agent_action(self, action: int):
        """
        Wendet eine Aktion des RL-Agenten auf den RECHTEN Schläger an.

        Aktionen:
          0 = Nichts tun
          1 = Hoch bewegen
          2 = Runter bewegen
          3 = Gegen Uhrzeigersinn kippen  (nur wenn tilting_enabled)
          4 = Im Uhrzeigersinn kippen     (nur wenn tilting_enabled)
        """
        self._move_paddle("agent", action)

    def apply_player_action(self, action: int):
        """Wendet eine Aktion des menschlichen Spielers auf den LINKEN Schläger an."""
        self._move_paddle("player", action)

    def apply_bot_action(self):
        """
        Einfacher regelbasierter Bot für den LINKEN Schläger (Trainingsgegner).

        Design-Überlegung:
          → Bot soll beatable sein, damit der Agent positive Belohnungen erhält
          → Bot soll nicht trivial sein, damit der Agent eine echte Strategie lernt
          → BOT_DIFFICULTY in config.py steuert die Schwierigkeit
        """
        bot_speed = PADDLE_SPEED * BOT_DIFFICULTY
        dead_zone = 5  # Pixel – Bot reagiert nicht auf kleine Abweichungen
        #self.player_angle=0

        diff = self.ball_y - self.player_y

        if diff > dead_zone:
            self.player_y = min(
                WINDOW_HEIGHT - PADDLE_HEIGHT / 2,
                self.player_y + bot_speed
            )
        elif diff < -dead_zone:
            self.player_y = max(
                PADDLE_HEIGHT / 2,
                self.player_y - bot_speed
            )
        
        if self.new_hit==True:
            if np.random.random() < 0.5*(BOT_DIFFICULTY**2):   # Chance für Bot das er die Schlägerstellung variiert
                self.player_angle = np.random.uniform(-PADDLE_MAX_TILT, PADDLE_MAX_TILT)*min(BOT_DIFFICULTY**2,1)
            self.new_hit=False

    def _move_paddle(self, who: str, action: int):
        """Gemeinsame Logik für Schläger-Bewegung."""
        if who == "agent":
            y_attr, angle_attr = "agent_y", "agent_angle"
        else:
            y_attr, angle_attr = "player_y", "player_angle"

        y     = getattr(self, y_attr)
        angle = getattr(self, angle_attr)

        if action == self.ACTION_UP:
            y = max(PADDLE_HEIGHT / 2, y - PADDLE_SPEED)

        elif action == self.ACTION_DOWN:
            y = min(WINDOW_HEIGHT - PADDLE_HEIGHT / 2, y + PADDLE_SPEED)

        elif action == self.ACTION_TILT_CCW and self.tilting_enabled:
            angle = max(-PADDLE_MAX_TILT, angle - PADDLE_TILT_SPEED)

        elif action == self.ACTION_TILT_CW and self.tilting_enabled:
            angle = min(PADDLE_MAX_TILT, angle + PADDLE_TILT_SPEED)

        setattr(self, y_attr,     y)
        setattr(self, angle_attr, angle)

    def move_wall(self):
        if self.wall_y<=WALL_HEIGHT/2:
            self.wall_moving_direction*=-1
        
        elif self.wall_y>=WINDOW_HEIGHT - WALL_HEIGHT / 2:
            self.wall_moving_direction*=-1
        
        self.wall_y+=WALL_SPEED*self.wall_moving_direction

    # ──────────────────────────────────────────────────────────
    #  SPIELSCHRITT (Physik-Update)
    # ──────────────────────────────────────────────────────────

    def step_physics(self) -> float:
        """
        Bewegt den Ball einen Frame weiter und löst alle Kollisionen auf.

        Returns:
        !GEÄNDERT!
          reward (float):
            +1.0  → Agent hat einen Punkt gemacht
            -1.0  → Gegner hat einen Punkt gemacht
            +0.2 → Agent hat den Ball getroffen (kleine Zwischen-Belohnung)
             0.0  → Kein besonderes Ereignis
        """
        # Ball bewegen
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        self.steps  += 1
        self.move_wall()

        return self._handle_collisions()

    def _handle_collisions(self) -> float:
        """
        Behandelt alle Kollisionen in der richtigen Reihenfolge.

        BELOHNUNGSDESIGN (Reward Shaping):
        ────────────────────────────────────
        Die Belohnungsstruktur ist einer der wichtigsten Aspekte in RL!
        Schlecht gewählte Belohnungen führen zu merkwürdigen Verhaltensweisen.

        Hier:
        !GEÄNDERT!
          +1.0  → Hauptziel (Punkt machen)
          -1.0  → Hauptvermeidung (Punkt kassieren)
          +0.2 → Zwischenziel (Ball überhaupt treffen)
                  → Hilft dem Agenten in frühen Trainingsphasen zu lernen
                  → Vorsicht: zu hohe Zwischen-Belohnung → Agent priorisiert
                    Treffer statt Punkte!
        """
        reward = 0.0

        # ── Wände oben und unten ──────────────────────────────
        if self.ball_y - BALL_SIZE / 2 <= 0:
            self.ball_y  = BALL_SIZE / 2
            self.ball_vy = abs(self.ball_vy)   # nach unten umkehren

        elif self.ball_y + BALL_SIZE / 2 >= WINDOW_HEIGHT:
            self.ball_y  = WINDOW_HEIGHT - BALL_SIZE / 2
            self.ball_vy = -abs(self.ball_vy)  # nach oben umkehren

        # ── Agent-Schläger (rechts) ───────────────────────────
        ax = AGENT_PADDLE_X
        ball_hits_agent = (
            self.ball_vx > 0 and   # Ball bewegt sich nach rechts
            self.ball_x + BALL_SIZE / 2 >= ax - PADDLE_WIDTH / 2 and
            self.ball_x - BALL_SIZE / 2 <= ax + PADDLE_WIDTH / 2 and
            abs(self.ball_y - self.agent_y) <= PADDLE_HEIGHT / 2 + BALL_SIZE / 2
        )
        ball_hits_wall = (
            # Ball erreicht die Wand von links ODER rechts
            self.ball_x + BALL_SIZE / 2 >= self.wall_x - WALL_WIDTH / 2 and
            self.ball_x - BALL_SIZE / 2 <= self.wall_x + WALL_WIDTH / 2 and

            # Vertikal innerhalb der Wand
            abs(self.ball_y - self.wall_y) <= WALL_HEIGHT / 2 + BALL_SIZE / 2
        )
        if ball_hits_agent:
            if self.last_hit=="" or self.last_hit=="player":
                self._reflect_ball(is_right_paddle=True,
                                paddle_angle=self.agent_angle,
                                paddle_x=ax)
                reward += 0.05   # Kleiner Bonus für Treffer
                self.last_hit="agent"

        if ball_hits_wall:
            new_state=""
            if  self.last_hit=="player":
                self._reflect_ball(is_right_paddle=True,paddle_angle=0,paddle_x=self.wall_x)
                new_state="agent"
            elif  self.last_hit=="agent":
                reward-=0.1
                self._reflect_ball(is_right_paddle=False,paddle_angle=0,paddle_x=self.wall_x)
                new_state="player"
            
            self.last_hit=new_state

        # Kleiner kontinuierlicher Reward für gute Y-Positionierung
        # wenn Ball auf Agenten zufliegt
        if self.ball_vx > 0:
            # Nähe des Balls zum Schläger (0 = weit weg, 1 = direkt davor)
            ball_proximity = self.ball_x / WINDOW_WIDTH  # 0 links, 1 rechts

            dist = abs(self.ball_y - self.agent_y) / (WINDOW_HEIGHT / 2)
            positioning = max(0, 1.0 - dist)

            reward += 0.0005 * positioning * ball_proximity

        # ── Spieler-Schläger (links) ──────────────────────────
        px = PLAYER_PADDLE_X
        ball_hits_player = (
            self.ball_vx < 0 and   # Ball bewegt sich nach links
            self.ball_x - BALL_SIZE / 2 <= px + PADDLE_WIDTH / 2 and
            self.ball_x + BALL_SIZE / 2 >= px - PADDLE_WIDTH / 2 and
            abs(self.ball_y - self.player_y) <= PADDLE_HEIGHT / 2 + BALL_SIZE / 2
        )
        if ball_hits_player:
            if self.last_hit=="" or self.last_hit=="agent":
                self._reflect_ball(is_right_paddle=False,
                                paddle_angle=self.player_angle,
                                paddle_x=px)
                self.last_hit="player"
                self.new_hit=True

        # ── Punkt gemacht? ────────────────────────────────────
        if self.ball_x > WINDOW_WIDTH + BALL_SIZE:
            reward = -2.5   # Ball hinter Agent → Spieler/Bot punktet

        elif self.ball_x < -BALL_SIZE:
            reward = 2.5    # Ball hinter Spieler → Agent punktet

        return reward

    # ──────────────────────────────────────────────────────────
    #  REFLEXIONSPHYSIK
    # ──────────────────────────────────────────────────────────

    def _reflect_ball(self, is_right_paddle: bool,
                      paddle_angle: float, paddle_x: float, paddle_width=PADDLE_WIDTH):
        """
        Berechnet die Reflexion des Balls an einem Schläger.

        ┌─────────────────────────────────────────────────────┐
        │  PHYSIK DER REFLEXION AN EINER OBERFLÄCHE           │
        │                                                     │
        │  Gegeben: Eingangsvektor v, Normalenvektor n        │
        │  Gesuchter Reflexionsvektor v':                     │
        │                                                     │
        │    v' = v − 2·(v·n)·n                              │
        │                                                     │
        │  Dabei ist n ein EINHEITS-Normalvektor.             │
        │                                                     │
        │  Für einen SENKRECHTEN Schläger:                    │
        │    Linker Schläger:  n = (1, 0)  → dreht vx um     │
        │    Rechter Schläger: n = (−1, 0) → dreht vx um     │
        │                                                     │
        │  Für einen GEKIPPTEN Schläger (Winkel θ):           │
        │    Der Normalvektor dreht sich ebenfalls um θ!      │
        │    Das ändert die Abprallrichtung des Balls.        │
        │                                                     │
        │  Beispiel: Rechter Schläger, θ = +30° (CW):        │
        │    n = (−cos30°, sin30°) ≈ (−0.87, 0.5)           │
        │    Ball wird nach oben-links abgelenkt              │
        └─────────────────────────────────────────────────────┘

        Args:
            is_right_paddle: True für Agent (rechts), False für Spieler (links)
            paddle_angle:    Kippwinkel in Grad (0 = senkrecht)
            paddle_x:        X-Position des Schläger-Zentrums
        """
        theta = math.radians(paddle_angle)

        # Normalvektor der Schlägeroberfläche berechnen
        if is_right_paddle:
            # Grundnormale zeigt nach links (−x), Kippung rotiert diesen Vektor
            nx =  -math.cos(theta)
            ny =   math.sin(theta)
        else:
            # Grundnormale zeigt nach rechts (+x), Kippung rotiert diesen Vektor
            nx =  math.cos(theta)
            ny =  math.sin(theta)

        # Reflexionsformel anwenden: v' = v − 2·(v·n)·n
        dot = self.ball_vx * nx + self.ball_vy * ny
        self.ball_vx = self.ball_vx - 2 * dot * nx
        self.ball_vy = self.ball_vy - 2 * dot * ny

        # Geschwindigkeit leicht erhöhen (dynamischeres Spiel)
        # aber nach oben begrenzen
        speed     = math.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if speed > 0:
            new_speed    = min(speed * 1.04, BALL_MAX_SPEED_FACTOR*BALL_INITIAL_SPEED)
            scale        = new_speed / speed
            self.ball_vx *= scale
            self.ball_vy *= scale

        # Ball aus dem Schläger herausschieben (verhindert "Stecken")
        clearance = paddle_width / 2 + BALL_SIZE / 2 + 1
        if is_right_paddle:
            self.ball_x = paddle_x - clearance
        else:
            self.ball_x = paddle_x + clearance

        #Korrektur falls Reflektion nicht gut funktioniert hat
        if is_right_paddle:
            if self.ball_vx > 0:
                self.ball_vx = -abs(self.ball_vx)
        else:
            if self.ball_vx < 0:
                self.ball_vx = abs(self.ball_vx)

    # ──────────────────────────────────────────────────────────
    #  BEOBACHTUNG FÜR DAS NEURONALE NETZ
    # ──────────────────────────────────────────────────────────

    def get_observation(self) -> np.ndarray:
        """
        Erstellt den Beobachtungsvektor für den RL-Agenten.

        NORMALISIERUNG ist wichtig für neuronale Netze!
        ─────────────────────────────────────────────────
        Neuronale Netze lernen besser, wenn alle Eingaben in einem
        ähnlichen Wertebereich liegen (hier: ca. [-1, 1]).

        Ohne Normalisierung: ball_x ∈ [0, 800], agent_y ∈ [0, 600]
        → Das Netz muss erst lernen, welche Zahlen "groß" oder "klein" sind

        Mit Normalisierung: alle Werte ∈ [-1, 1]
        → Das Netz kann sofort mit vernünftigen Gewichten starten

        Beobachtungsvektor (7 oder 9 Werte):
          [0] ball_x_norm         Ball-Position horizontal
          [1] ball_y_norm         Ball-Position vertikal
          [2] ball_vx_norm        Ball-Geschwindigkeit horizontal
          [3] ball_vy_norm        Ball-Geschwindigkeit vertikal
          [4] agent_y_norm        Agent-Schläger Position
          [5] player_y_norm       Gegner-Schläger Position
          [6] Wall_y              Position der Wand vertikal
          [7] agent_angle_norm    Agent-Kippwinkel    (nur wenn tilting)
          [8] player_angle_norm   Gegner-Kippwinkel   (nur wenn tilting)

        """
        obs = [
            (self.ball_x  / WINDOW_WIDTH)  * 2 - 1,           # [-1, 1]
            (self.ball_y  / WINDOW_HEIGHT) * 2 - 1,           # [-1, 1]
            self.ball_vx / (BALL_MAX_SPEED_FACTOR*BALL_INITIAL_SPEED),  # Maximalgeschwindigkeit als Nenner
            self.ball_vy / (BALL_MAX_SPEED_FACTOR*BALL_INITIAL_SPEED),  # → exakt [-1, 1] ohne clip()
            (self.agent_y  / WINDOW_HEIGHT) * 2 - 1,          # [-1, 1]
            (self.player_y / WINDOW_HEIGHT) * 2 - 1,          # [-1, 1]
            (self.wall_y / WINDOW_HEIGHT)*2 -1
        ]

        if self.tilting_enabled:
            obs += [
                self.agent_angle  / PADDLE_MAX_TILT,           # [-1, 1]
                self.player_angle / PADDLE_MAX_TILT,           # [-1, 1]
            ]

        return np.array(obs, dtype=np.float32)

    # ──────────────────────────────────────────────────────────
    #  HILFSMETHODEN
    # ──────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Dimension des Beobachtungsvektors."""
        return 9 if self.tilting_enabled else 7

    @property
    def n_actions(self) -> int:
        """Anzahl verfügbarer Aktionen."""
        return 5 if self.tilting_enabled else 3

    def is_done(self, reward: float) -> bool:
        """True wenn die Episode durch einen Punkt beendet wurde."""
        return abs(reward) >= 1.0
