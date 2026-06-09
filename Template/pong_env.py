"""
╔══════════════════════════════════════════════════════════════╗
║       PONG RL  ·  Gymnasium-Umgebung (pong_env.py)          ║
╚══════════════════════════════════════════════════════════════╝

WAS IST EINE GYMNASIUM-UMGEBUNG?
──────────────────────────────────
Gymnasium (früher: OpenAI Gym) definiert eine STANDARD-SCHNITTSTELLE
für RL-Umgebungen. Jede Umgebung implementiert:

  env.reset()        → Neues Spiel starten, ersten Zustand zurückgeben
  env.step(action)   → Einen Schritt ausführen
                       → gibt (zustand, belohnung, fertig, info) zurück
  env.render()       → Spiel visualisieren
  env.close()        → Aufräumen

Warum ein Standard? Damit RL-Algorithmen (z.B. PPO) mit JEDER
Umgebung funktionieren, die das Interface implementiert.
Das ist wie USB: gleicher Stecker, verschiedene Geräte.

AGENT vs. UMGEBUNG:

    ┌─────────────────────────────────────────────┐
    │                                             │
    │   Zustand s ──────────────→ Agent (PPO)     │
    │                                             │
    │   Aktion a  ←────────────── Agent (PPO)     │
    │         │                                   │
    │         ▼                                   │
    │   Umgebung (Pong)                           │
    │         │                                   │
    │         ▼                                   │
    │   Belohnung r + neuer Zustand s'            │
    │                                             │
    └─────────────────────────────────────────────┘

Der Agent sieht nur den Zustand und die Belohnung.
Er kennt die Spielregeln nicht – er LERNT sie!
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from config import *
from pong_game import PongGame

import sys
sys.stdout.reconfigure(encoding='utf-8')


class PongEnv(gym.Env):
    """
    Pong als Gymnasium-kompatible RL-Umgebung.

    Der RL-AGENT spielt auf der RECHTEN Seite (rot).
    Der TRAININGSGEGNER (einfacher Bot) spielt auf der LINKEN Seite (blau).

    Erbt von gym.Env → implementiert das Standard-Interface.
    stable-baselines3's PPO erwartet genau dieses Interface.
    """

    # Metadaten: Welche Render-Modi werden unterstützt?
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, tilting_enabled=TILTING_ENABLED):
        super().__init__()

        self.render_mode      = render_mode
        self.tilting_enabled  = tilting_enabled

        # Interne Spielinstanz (aus pong_game.py)
        self.game = PongGame(tilting_enabled=tilting_enabled)

        # ── AKTIONSRAUM ──────────────────────────────────────
        # Discrete(n): Agent wählt eine INTEGER aus {0, 1, ..., n-1}
        #
        # Ohne Kippen: n=3   → {0=Nichts, 1=Hoch, 2=Runter}
        # Mit Kippen:  n=5   → {0=Nichts, 1=Hoch, 2=Runter, 3=CCW, 4=CW}
        #
        # Alternativ gäbe es: spaces.Box (kontinuierlich) oder
        # spaces.MultiDiscrete (mehrere unabhängige diskrete Aktionen)
        self.action_space = spaces.Discrete(self.game.n_actions)

        # ── BEOBACHTUNGSRAUM ─────────────────────────────────
        # Box: n-dimensionaler Raum mit lower/upper bounds
        # Alle Werte normalisiert → ungefähr [-1, 1]
        # (Clip sorgt dafür, dass Grenzwerte wirklich eingehalten werden)
        obs_dim = self.game.obs_dim
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -1.5, dtype=np.float32),
            high=np.full(obs_dim,  1.5, dtype=np.float32),
        )

        # Pygame wird nur bei render_mode="human" initialisiert
        self.screen = None
        self.clock  = None
        self._font_large = None
        self._font_small = None

        # Interne Statistik
        self._episode_reward = 0.0
        self._max_steps      = 10000   # Zeitlimit pro Episode

    # ──────────────────────────────────────────────────────────
    #  GYMNASIUM INTERFACE – Diese 4 Methoden MÜSSEN implementiert sein
    # ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Startet eine neue Episode (ein neues Spiel).

        Wird aufgerufen:
          • Zu Beginn des Trainings
          • Nach jedem Punkt (terminated=True)
          • Wenn Zeitlimit erreicht (truncated=True)

        Returns:
          observation (np.ndarray): Der Anfangszustand
          info (dict): Optionale Zusatzinfos
        """
        super().reset(seed=seed)   # Setzt np.random Seed

        self.game.reset()
        self._episode_reward = 0.0

        if self.render_mode == "human":
            self._setup_pygame()

        return self.game.get_observation(), {}

    def step(self, action: int):
        """
        Führt EINEN Spielschritt aus.

        Das ist die Kernfunktion des RL-Lernens.
        Sie wird tausende Male pro Sekunde aufgerufen!

        Args:
            action (int): Die vom Agenten gewählte Aktion

        Returns:
            observation (np.ndarray) : Neuer Spielzustand
            reward      (float)      : Belohnung (Lernsignal!)
            terminated  (bool)       : Normales Spielende (Punkt)
            truncated   (bool)       : Zeitlimit erreicht
            info        (dict)       : Zusatzinfos für Debugging
        """
        # 1. Agent-Aktion auf rechten Schläger anwenden
        self.game.apply_agent_action(action)

        # 2. Bot-Aktion auf linken Schläger anwenden
        self.game.apply_bot_action()

        # 3. Physik-Update (Ball bewegen, Kollisionen)
        reward = self.game.step_physics()

        # 4. Episode-Ende prüfen
        terminated = self.game.is_done(reward)          # Punkt gemacht
        truncated  = self.game.steps >= self._max_steps  # Zeitlimit

        self._episode_reward += reward

        # 5. Optional rendern
        if self.render_mode == "human":
            self._render_frame()

        info = {
            "steps":          self.game.steps,
            "episode_reward": self._episode_reward,
            "agent_scored":   reward >= 1.0,    # True nur wenn Punkt gemacht
            "bot_scored":     reward <= -1.0,   # True nur wenn Punkt kassiert
            "TimeLimit.truncated": truncated,
        }

        return self.game.get_observation(), reward, terminated, truncated, info

    def render(self):
        """Rendert das Spiel (wird von Gymnasium aufgerufen)."""
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._get_rgb_array()

    def close(self):
        """Gibt Ressourcen frei."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # ──────────────────────────────────────────────────────────
    #  PYGAME RENDERING
    # ──────────────────────────────────────────────────────────

    def _setup_pygame(self):
        """Initialisiert Pygame (nur beim ersten Aufruf)."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Pong RL – Training")
            self.clock       = pygame.time.Clock()
            self._font_large = pygame.font.Font(None, 72)
            self._font_small = pygame.font.Font(None, 26)

    def _render_frame(self):
        """Zeichnet einen Frame."""
        if self.screen is None:
            self._setup_pygame()

        g = self.game   # Abkürzung

        # Events verarbeiten (damit das Fenster reagiert)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Hintergrund
        self.screen.fill((15, 15, 20))

        # Mittellinie (gestrichelt)
        for y in range(0, WINDOW_HEIGHT, 25):
            pygame.draw.rect(self.screen, (55, 55, 65),
                             (WINDOW_WIDTH // 2 - 2, y, 4, 14))

        # Ball
        pygame.draw.circle(
            self.screen, (240, 240, 255),
            (int(g.ball_x), int(g.ball_y)),
            BALL_SIZE // 2
        )

        # Schläger zeichnen
        self._draw_paddle(PLAYER_PADDLE_X, g.player_y, g.player_angle, (80, 160, 255))
        self._draw_paddle(AGENT_PADDLE_X,  g.agent_y,  g.agent_angle,  (255, 80,  80))

        # Info-Leiste unten
        tilt_str = "Kippen: AN" if self.tilting_enabled else "Kippen: AUS"
        info_txt = self._font_small.render(
            f"{tilt_str}  |  Schritt: {g.steps}  |  Belohnung: {self._episode_reward:+.2f}",
            True, (90, 90, 110)
        )
        self.screen.blit(info_txt, (10, WINDOW_HEIGHT - 28))

        # Labels
        p_lbl = self._font_small.render("BOT", True, (80, 160, 255))
        a_lbl = self._font_small.render("AGENT", True, (255, 80, 80))
        self.screen.blit(p_lbl, (WINDOW_WIDTH // 4 - 18, 20))
        self.screen.blit(a_lbl, (3 * WINDOW_WIDTH // 4 - 28, 20))

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_paddle(self, x: int, y: float, angle_deg: float, color: tuple):
        """
        Zeichnet einen Schläger mit optionaler Rotation.
        pygame.transform.rotate() dreht eine Surface um ihren Mittelpunkt.
        """
        surf = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(surf, color, (0, 0, PADDLE_WIDTH, PADDLE_HEIGHT),
                         border_radius=4)

        if abs(angle_deg) > 0.5:
            # Negatives Vorzeichen: Pygame dreht CCW für positive Winkel,
            # wir wollen CW für positive angle_deg
            surf = pygame.transform.rotate(surf, -angle_deg)

        rect = surf.get_rect(center=(int(x), int(y)))
        self.screen.blit(surf, rect)

    def _get_rgb_array(self) -> np.ndarray:
        """Gibt den aktuellen Frame als RGB-Array zurück."""
        if self.screen is None:
            self._setup_pygame()
            self._render_frame()
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
