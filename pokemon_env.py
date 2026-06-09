"""
╔══════════════════════════════════════════════════════════════╗
║  POKEMON GOLD RL  ·  Gymnasium-Umgebung (pokemon_env.py)    ║
╚══════════════════════════════════════════════════════════════╝

WIE UNTERSCHEIDET SICH DIESE UMGEBUNG VON PONG?
──────────────────────────────────────────────────
Pong war einfach:
  • Klarer Zustand (7 Zahlen)
  • Eindeutiges Belohnungssignal (+1 Punkt, -1 Punkt)
  • Episode endet nach jedem Punkt

Pokemon ist VIEL komplexer:
  • Bild-Input (CNN notwendig)
  • Sehr SPARSE Rewards (Belohnungen kommen selten!)
    → Die KI kann 1000 Schritte laufen ohne einen Reward zu sehen
  • Langfristige Ziele (z.B. Pokemon fangen dauert viele Schritte)
  • Viele verschiedene Spielzustände (Karte, Kampf, Menü, Dialog...)

BEOBACHTUNGSRAUM (MultiInputPolicy):
  Wir geben der KI ZWEI verschiedene Inputs gleichzeitig:

  1. "screen" → 84×84 Graustufenbild (CNN verarbeitet dies)
     ┌──────────────────────────────────────┐
     │  Das visuelle Bild des GameBoy       │
     │  → KI sieht Bewegungen, Gegner, etc. │
     └──────────────────────────────────────┘

  2. "ram_features" → 10 normalisierte Zahlen (MLP verarbeitet dies)
     ┌──────────────────────────────────────┐
     │  Position, HP, Pokemon-Anzahl, etc.  │
     │  → KI weiß exakt wo sie ist          │
     └──────────────────────────────────────┘

  Stable-Baselines3 kombiniert beide Inputs automatisch
  wenn wir MultiInputPolicy verwenden!

REWARDS (Belohnungssystem):
  Das Belohnungssystem ist AUSKOMMENTIERT – du sollst es selbst ausarbeiten!
  Hier sind nur Grundstrukturen angelegt. Die Kunst beim RL ist es,
  gute Rewards zu definieren: zu dichte Rewards → KI "cheated",
  zu sparse Rewards → KI lernt nichts.

  Tipps für Pokemon:
  • Belohne neue Positionen (Exploration fördern)
  • Belohne das Fangen des ersten Pokemon (großer Bonus)
  • Bestrafe HP-Verlust (KI soll nicht rücksichtslos kämpfen)
  • Belohne Fortschritt (neue Karte betreten, Dialog beenden)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2   # OpenCV: Bildverarbeitung (Skalierung, Graustufen)
import io
from collections import deque

from pyboy import PyBoy

from config import *
from ram_reader import (
    get_all_ram_features,
    read_player_position,
    read_hp,
    read_enemy_hp,
    read_party_count,
    read_party_species,
    read_caught_pokemon_count,
    read_badge_count,
    read_exp,
    read_total_party_xp,
    read_level,
    read_money,
    read_in_battle,
    print_game_state,
)
from global_coords import GlobalCoordinateTransform

import sys
sys.stdout.reconfigure(encoding='utf-8')


class PokemonGoldEnv(gym.Env):
    """
    Pokemon Gold als Gymnasium-kompatible RL-Umgebung.

    Die KI spielt Pokemon Gold über PyBoy (GameBoy-Emulator).
    Sie sieht das Spielbild (CNN) + RAM-Werte (MLP) und wählt
    in jedem Schritt eine Taste zum Drücken.

    Diese Klasse erbt von gym.Env und implementiert:
      reset()  → Spiel neu starten / Save-State laden
      step()   → Taste drücken, Frames simulieren, Reward berechnen
      render() → Aktuelles Bild anzeigen (optional)
      close()  → PyBoy sauber beenden

    Verwendung:
        env = PokemonGoldEnv()
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)  # 'up' drücken
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, record_actions=False, verbose=True):
        """
        Initialisiert die Umgebung.

        Args:
            render_mode    : "rgb_array" zum Rendern, None für headless Training
            record_actions : True = speichert alle Aktionen für späteres Replay
            verbose        : False = keine Konsolenausgabe (für Trainings-Subprozesse)
        """
        super().__init__()

        self.render_mode    = render_mode
        self.record_actions = record_actions
        self.verbose        = verbose

        # Liste aller Aktionen (aus config.py)
        # Der Index entspricht der Zahl die das Netz ausgibt:
        # 0='up', 1='down', 2='left', 3='right', 4='a', 5='b', 6='start'
        self.actions = ACTIONS
        self.n_actions = len(self.actions)

        # ── AKTIONSRAUM ──────────────────────────────────────
        # Discrete(n): KI wählt eine Ganzzahl aus {0, 1, ..., n-1}
        self.action_space = spaces.Discrete(self.n_actions)

        # ── BEOBACHTUNGSRAUM (Dict) ──────────────────────────
        # MultiInputPolicy erwartet ein Dictionary mit mehreren Inputs.
        # Jeder Eintrag ist ein eigener Beobachtungsraum (gym.spaces).
        #
        # "screen": Graustufen-Bild als 2D-Array mit einem Kanal
        #   Shape: (SCREEN_HEIGHT, SCREEN_WIDTH, 1)
        #   Dtype: uint8 (0-255) → CNN-Standard
        #
        # "ram_features": Normalisierte RAM-Werte
        #   Shape: (N_RAM_FEATURES,)
        #   Dtype: float32 ([0.0, 1.0])
        screen_channels = 1 if GRAYSCALE else 3
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(
                low=0, high=255,
                shape=(SCREEN_HEIGHT, SCREEN_WIDTH, screen_channels),
                dtype=np.uint8
            ),
            "ram_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(N_RAM_FEATURES,),
                dtype=np.float32
            ),
        })

        # ── PYBOY EMULATOR ───────────────────────────────────
        # PyBoy ist der GameBoy-Emulator.
        # "headless" = kein Fenster öffnen (für Training)
        # "SDL2"     = Fenster öffnen (für visuelles Debugging)
        self.pyboy = None   # Wird in reset() initialisiert

        # Interner Zustand
        self._steps           = 0
        self._episode_reward  = 0.0
        self._episode_xp      = 0              # Gesammelte XP in dieser Episode
        self._last_action     = -1             # Zuletzt ausgeführte Aktion (für Reward-Berechnung)
        self._action_history  = []              # Für Replay aufzeichnen (falls record_actions=True)
        self._reward_history  = deque(maxlen=N_STEPS_WITHOUT_REWARD)  # Letzten N_STEPS_WITHOUT_REWARD Rewards für Early-Stop-Check

        # Zustandsverfolgung für Reward-Berechnung
        # (was war der Zustand im LETZTEN Schritt?)
        self._prev_position   = (0, 0, 0, 0)   # (map_group, map, x, y)
        self._prev_hp         = 0              # Eigene HP (Totodile)
        self._prev_max_hp     = 1
        self._prev_enemy_hp              = 0     # Gegner-HP im letzten Schritt
        self._prev_in_battle             = False # Kampfstatus im letzten Schritt
        self._battle_start_hp            = 0     # Gegner-HP zu Kampfbeginn (für Catchable-Zone)
        self._battle_min_enemy_hp        = 0     # Minimum der Gegner-HP im aktuellen Kampf
        self._enemy_defeated_this_battle = False # Gegner besiegt? (verhindert falsche Flucht-Strafe)
        self._prev_caught         = 0
        self._prev_party          = 0
        self._prev_XP             = 0
        self._prev_party_species  = set()          # Arten-IDs im Team (für Diversitäts-Reward)
        self._visited_positions   = set()          # Alle je besuchten Positionen
        self._visited_maps        = set()          # Karten-IDs die je betreten wurden (für Map-Bonus)
        self._maps_seen           = set()          # tatsächlich betretene Karten (ehrliche Anzeige)
        self._cleared_maps        = set()          # Karten-IDs mit reduziertem Reward (abgegrast)
        self._trap_maps           = set()          # Ablenkungsgebäude → kleiner negativer Reward
        self._initial_map_count   = 0              # Vorbekannte Karten beim Episode-Start (für saubere Anzeige)

        # Globale Weltkoordinaten-Transformation (eine Instanz pro Umgebung,
        # damit der Carry-Forward-Zustand nicht zwischen parallelen Envs kollidiert)
        self._gct                 = GlobalCoordinateTransform()

        # Frontier-Reward: globaler Startpunkt + größte bisher erreichte Distanz
        self._start_global        = (0, 0)
        self._max_dist_from_start = 0.0

        # Save-State-Bytes (werden in reset() zufällig gewählt)
        self._save_state_pool = []   # Liste aller geladenen States
        self._load_save_state()

    def _load_save_state(self):
        """
        Lädt alle Save-States aus SAVE_STATE_PATHS in den Speicher.

        Jeder State wird als Bytes-Objekt in self._save_state_pool abgelegt.
        In reset() wird dann zufällig einer gewählt (Curriculum Learning):
        Die KI startet jede Episode an einem anderen Punkt und lernt
        so alle Gebiete gleichzeitig, ohne frühere Bereiche zu vergessen.

        Neue Gebiete einfach in config.py → SAVE_STATE_PATHS ergänzen!
        """
        self._save_state_pool = []
        for path in SAVE_STATE_PATHS:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self._save_state_pool.append((path, f.read()))
                if self.verbose:
                    print(f"  ✓ Save-State geladen: {path}")
            else:
                if self.verbose:
                    print(f"  ⚠ Save-State nicht gefunden (übersprungen): {path}")

        if not self._save_state_pool:
            if self.verbose:
                print("  ⚠ Keine Save-States gefunden – Spiel startet vom ROM-Anfang.")
                print("    Tipp: Erstelle States mit create_savestate.py!")

    def _init_pyboy(self):
        """
        Erstellt eine neue PyBoy-Instanz.

        Wird bei jedem reset() aufgerufen wenn noch keine Instanz existiert.
        """
        if self.pyboy is not None:
            self.pyboy.stop()

        self.pyboy = PyBoy(
            ROM_PATH,
            window="null",   # "null" = kein Fenster (headless, für Training)
                              # "SDL2" = Fenster anzeigen (für Debugging)
        )

        # Emulationsgeschwindigkeit: 0 = so schnell wie möglich
        # 1 = Echtzeit (60 FPS), 2 = 2× Geschwindigkeit, usw.
        self.pyboy.set_emulation_speed(EMULATION_SPEED)

    # ──────────────────────────────────────────────────────────
    #  GYMNASIUM INTERFACE
    # ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Startet eine neue Episode.

        Bei jedem reset() wird der Save-State geladen → KI startet
        immer vom gleichen Punkt. Das ist wie F5 im Spiel!

        Returns:
            observation (dict) : {'screen': np.array, 'ram_features': np.array}
            info        (dict) : Zusatzinfos (Position, HP, etc.)
        """
        super().reset(seed=seed)

        # PyBoy initialisieren (beim ersten Mal oder nach close())
        if self.pyboy is None:
            self._init_pyboy()

        # Save-State laden – zufällig aus dem Pool wählen (Curriculum Learning)
        if self._save_state_pool:
            idx = int(self.np_random.integers(len(self._save_state_pool)))
            chosen_path, chosen_bytes = self._save_state_pool[idx]
            state_io = io.BytesIO(chosen_bytes)
            self.pyboy.load_state(state_io)
        else:
            # Kein State vorhanden: Intro läuft durch
            # HINWEIS: Das ist nicht ideal. Erstelle lieber einen Save-State!
            pass

        # Interne Zähler zurücksetzen
        self._steps          = 0
        self._episode_reward = 0.0
        self._episode_xp     = 0
        self._action_history = []
        self._reward_history = deque(maxlen=N_STEPS_WITHOUT_REWARD)
        self._visited_positions = set()
        self._gct.reset()                       # Carry-Forward der Weltkoordinaten leeren
        self._prev_enemy_hp              = 0
        self._prev_in_battle             = False
        self._battle_start_hp            = 0
        self._battle_min_enemy_hp        = 0
        self._enemy_defeated_this_battle = False
        self._prev_party_species = set()

        # Anfangszustand aus RAM lesen (für Reward-Differenzen im ersten Schritt)
        self._prev_position = read_player_position(self.pyboy)

        # Frontier-Reward: globalen Startpunkt setzen, Max-Distanz zurücksetzen.
        sb, sn, sx, sy            = self._prev_position
        start_g                   = self._gct.to_global(sb, sn, sx, sy)
        self._start_global        = start_g if start_g is not None else (0, 0)
        self._max_dist_from_start = 0.0

        hp, max_hp          = read_hp(self.pyboy)
        self._prev_hp       = hp
        self._prev_max_hp   = max(max_hp, 1)
        self._prev_caught   = read_caught_pokemon_count(self.pyboy)
        self._prev_party    = read_party_count(self.pyboy)
        self._prev_XP       = read_total_party_xp(self.pyboy)

        # ── CLEARED MAPS & VISITED MAPS vorinitialisieren ────────
        # Cleared maps: Route 29 + New Bark Town immer abgegrast (verifizierte IDs).
        # Visited maps: enthält von Anfang an ALLE cleared maps + die Startkarte.
        #   → cleared maps bekommen NIE den einmaligen Map-Entdeckungs-Bonus (+7),
        #     weder beim ersten Schritt noch bei späteren Episoden.
        #   → Die Startkarte auch nicht (kein Bonus für den Startpunkt).
        # Wichtig: cleared_maps muss hier stehen, NICHT in _calculate_reward()!
        # Ablenkungsgebäude: winzige Räume, KI verläuft sich dort nur.
        # Beim ERSTEN Betreten pro Episode: kleiner negativer Reward statt Map-Bonus.
        # Sind automatisch auch in cleared_maps (kein XP/Tile-Reward-Farming drin).
        self._trap_maps = {
            # New Bark Town Innenräume  ✓ verifiziert (Disassembly-Namen)
            (24, 5),   # Elms Labor
            (24, 6),   # Spielerhaus EG
            (24, 7),   # Spielerhaus OG – Schlupfloch wie Pokecenter-OG (20,1)
            (24, 8),   # Nachbarhaus
            (24, 9),   # Elms Haus
            # Cherrygrove Gebäude  ✓ verifiziert (KI kann dort nichts Sinnvolles lernen)
            (26, 4),   # Pokemart    – Einkaufen zu komplex für aktuelle KI
            (26, 5),   # Pokemon Center EG – Heilen zu komplex; Save State Ansatz stattdessen
            (20, 1),   # Pokemon Center OG – KI farmt dort den Map-Bonus ✗
            (26, 6), (26, 7), (26, 8),   # Wohnhäuser
            (26, 9),   # Haus auf Route 30 – KI läuft dort rein, kein Mehrwert
            (26, 10),  # Mr. Pokémons Haus – KI läuft dort rein, kein Mehrwert
        }
        self._cleared_maps  = {
            (24, 4),   # New Bark Town   ✓ verifiziert
            (24, 3),   # Route 29        ✓ verifiziert
            # (26, 5) = Pokemon Center → NICHT cleared, voller +7 Bonus (Heilen wichtig!)
        } | self._trap_maps   # Trap-Maps sind immer auch "abgegrast"
        map_group, map_number, _, _ = self._prev_position
        self._visited_maps  = self._cleared_maps | {(map_group, map_number)}
        self._initial_map_count = len(self._visited_maps)  # (nicht mehr für Anzeige genutzt)
        self._maps_seen     = {(map_group, map_number)}    # echte Zählung: Startkarte zählt mit
        self._prev_party_species = read_party_species(self.pyboy)

        obs  = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int):
        """
        Führt EINEN Schritt aus: Taste drücken + FRAMES_PER_ACTION Frames simulieren.

        Das ist das Herzstück des RL-Lernens!

        Args:
            action (int) : Index in self.actions (0='up', 1='down', ...)

        Returns:
            obs        (dict)  : Neuer Spielzustand (Bild + RAM)
            reward     (float) : Belohnung (dein Lernsignal!)
            terminated (bool)  : Episode normal beendet (z.B. KI besiegt)
            truncated  (bool)  : Zeitlimit erreicht (MAX_STEPS_PER_EPISODE)
            info       (dict)  : Debuginfos
        """
        # Aktion aufzeichnen (für späteren Replay)
        if self.record_actions:
            self._action_history.append(action)

        # ── 1. TASTE DRÜCKEN ─────────────────────────────────
        # button_str = z.B. "up", "a", "b"
        button_str = self.actions[action]

        # ── 2. FRAMES SIMULIEREN ─────────────────────────────
        # button_press hält die Taste für FRAMES_PER_ACTION Frames gedrückt.
        # 1 Tile Bewegung = ~16 Frames → 24 Frames gibt genug Puffer.
        # button_release danach damit die Taste nicht dauerhaft gedrückt bleibt.
        self.pyboy.button_press(button_str)
        for _ in range(FRAMES_PER_ACTION):
            self.pyboy.tick()
        self.pyboy.button_release(button_str)

        # ── 3. NEUEN ZUSTAND LESEN ───────────────────────────
        obs = self._get_observation()

        # ── 4. REWARD BERECHNEN ──────────────────────────────
        self._last_action = action
        reward = self._calculate_reward()

        # ── 5. EPISODENENDE PRÜFEN ───────────────────────────
        terminated = self._check_terminated()
        truncated  = self._steps >= MAX_STEPS_PER_EPISODE

        # Zähler aktualisieren
        self._steps          += 1
        self._episode_reward += reward

        info = self._get_info()

        # Vorherigen Zustand für nächsten Schritt merken
        self._update_prev_state()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Gibt den aktuellen Frame als RGB-Array zurück."""
        if self.render_mode == "rgb_array":
            return self._get_screen_rgb()

    def close(self):
        """Beendet den Emulator sauber."""
        if self.pyboy is not None:
            self.pyboy.stop()
            self.pyboy = None

    # ──────────────────────────────────────────────────────────
    #  BEOBACHTUNG (was sieht die KI?)
    # ──────────────────────────────────────────────────────────

    def _get_screen(self) -> np.ndarray:
        """
        Liest den aktuellen GameBoy-Bildschirm und bereitet ihn für das CNN vor.

        GameBoy Color Originalauflösung: 160×144 Pixel, RGBA (4 Kanäle)
        Ausgabe: SCREEN_HEIGHT × SCREEN_WIDTH × 1 (Graustufen, uint8)

        Schritte:
        1. Bildschirm als numpy-Array lesen
        2. RGBA → Graustufen konvertieren (cv2.cvtColor)
        3. Auf 84×84 skalieren (cv2.resize)
        4. Dimension hinzufügen für CNN: (84, 84) → (84, 84, 1)
        """
        # PyBoy gibt RGBA-Bild zurück: Shape (144, 160, 4), dtype uint8
        screen_rgba = self.pyboy.screen.ndarray

        if GRAYSCALE:
            # RGBA → RGB → Graustufen
            screen_rgb  = cv2.cvtColor(screen_rgba, cv2.COLOR_RGBA2RGB)
            screen_gray = cv2.cvtColor(screen_rgb,  cv2.COLOR_RGB2GRAY)
            # Auf Zielpixelgröße skalieren (SCREEN_HEIGHT × SCREEN_WIDTH)
            screen_resized = cv2.resize(
                screen_gray,
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                interpolation=cv2.INTER_AREA   # INTER_AREA ist gut für Verkleinerung
            )
            # Dimension hinzufügen: (84, 84) → (84, 84, 1) [CNN erwartet Kanal-Dim]
            return screen_resized[:, :, np.newaxis]
        else:
            screen_rgb = cv2.cvtColor(screen_rgba, cv2.COLOR_RGBA2RGB)
            screen_resized = cv2.resize(
                screen_rgb,
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            return screen_resized   # (84, 84, 3)

    def _get_screen_rgb(self) -> np.ndarray:
        """Gibt das Bild als RGB für Rendering zurück (ungekürzt)."""
        screen_rgba = self.pyboy.screen.ndarray
        return cv2.cvtColor(screen_rgba, cv2.COLOR_RGBA2RGB)

    def _get_observation(self) -> dict:
        """
        Erstellt die vollständige Beobachtung für die KI.

        Gibt ein Dictionary zurück das MultiInputPolicy erwartet.
        """
        return {
            "screen":       self._get_screen(),
            "ram_features": get_all_ram_features(self.pyboy, self._gct),
        }

    # ──────────────────────────────────────────────────────────
    #  REWARD-BERECHNUNG  ← DAS IST DEIN PART!
    # ──────────────────────────────────────────────────────────

    def _calculate_reward(self) -> float:
        """
        Berechnet den Reward für den aktuellen Schritt.

        ════════════════════════════════════════════════════════
        REWARD SHAPING – das wichtigste Konzept für dich!
        ════════════════════════════════════════════════════════

        Reward Shaping bedeutet: du designst das Belohnungssignal
        so, dass die KI das tut was du willst.

        Wichtige Prinzipien:
        ─────────────────────
        1. SPARSE vs. DENSE REWARDS:
           • Sparse: nur bei wichtigen Ereignissen (Pokemon fangen = +10)
             → KI muss viel erkunden, lernt langsam aber generalisiert besser
           • Dense: bei jedem Schritt kleine Belohnungen (neue Position = +0.01)
             → KI lernt schneller, kann aber "cheaten" (z.B. im Kreis laufen)

        2. REWARD MAGNITUDE:
           • Halte Rewards in einer sinnvollen Größenordnung (z.B. -1 bis +10)
           • Zu große Unterschiede destabilisieren das Training

        3. DICHTE UND SPARSE KOMBINIEREN:
           • Kleine dichte Rewards für Orientierung (Exploration)
           • Große sparse Rewards für echte Ziele (Pokemon fangen)

        4. BESTRAFUNGEN SPARSAM EINSETZEN:
           • Zu viele Bestrafungen → KI wird passiv (tut nichts um Strafe zu vermeiden!)
           • Besser: fehlender Fortschritt = negativer Reward

        ════════════════════════════════════════════════════════

        TODO: Passe diese Funktion nach deinen Vorstellungen an!
        Die aktuellen Beispiele sind ein Startpunkt, kein Endpunkt.
        """
        reward = 0.0

        # ── AKTUELLE ZUSTÄNDE AUSLESEN ───────────────────────
        map_group, map_number, x, y = read_player_position(self.pyboy)
        current_hp, max_hp          = read_hp(self.pyboy)
        party_count                 = read_party_count(self.pyboy)
        party_species               = read_party_species(self.pyboy)
        caught_count                = read_caught_pokemon_count(self.pyboy)
        XP_count                    = read_total_party_xp(self.pyboy)
        enemy_hp                    = read_enemy_hp(self.pyboy)
        in_battle                   = read_in_battle(self.pyboy)
        current_level               = read_level(self.pyboy)

        map_key                     = (map_group, map_number)
        self._maps_seen.add(map_key)   # ehrliche Zählung jeder tatsächlich betretenen Karte

        # Cleared maps bekommen einen dynamisch sinkenden Reward-Faktor: 4 / Level.
        # Verifizierte Karten-IDs (debug_ram.py Walk-Through):
        #   New Bark Town : (24, 4)  → immer cleared
        #   Route 29      : (24, 3)  → immer cleared
        #   Cherrygrove   : (26, 3)  → voller Reward (Stadt, kein XP-Grind möglich)
        #   Pokemart      : (26, 4)  → Trap-Map
        #   Route 30      : (26, 1)  → cleared ab Level 13 (langer Korridor)
        #   Route 31      : (26, 2)  → NICHT clearen! Weg nach Violet City
        # → Level  9: Faktor ≈ 0.33  (30% Reward, wie früher beim harten Cutoff)
        # → Level 15: Faktor = 0.20  (steigender Druck weiterzuziehen)
        # → Level 30: Faktor = 0.10  (kaum noch Anreiz zu bleiben)
        # min(1.0, ...) verhindert Faktor > 1 bei sehr niedrigem Level (4/1 = 4.0).
        # Neue, unbekannte Maps bekommen immer VOLLEN Reward → KI wird aktiv hingelockt.

        # Dynamisch: Route 30 (beide Teile) wird cleared sobald Level 13 erreicht.
        # Set.add() ist idempotent → kein Problem wenn das jeden Schritt läuft.
        if current_level >= 13:
            self._cleared_maps.add((26, 1))   # Route 30   ✓ verifiziert
            #self._cleared_maps.add((26, 2))   # Route 31    ✓ verifiziert
 
        if map_key in self._cleared_maps:
            factor = min(1.0, 4.0 / max(current_level, 1))
        else:
            factor = 1.0   # Neue Maps: volle Belohnung → KI soll dorthin!

        # ── BEISPIEL 1: EXPLORATION REWARD ───────────────────
        # Belohne das Betreten einer neuen Position.
        # Das motiviert die KI zu erkunden statt auf der Stelle zu stehen.
        #
        # Wir speichern jede besuchte (map, x, y) als "visited".
        # Neue Position → kleiner Bonus!

        current_pos_key = (map_group, map_number, x, y)
        if current_pos_key not in self._visited_positions:
            self._visited_positions.add(current_pos_key)
            # reward += 0.03 * factor   # AUS: ersetzt durch Frontier-Reward (max. Distanz).
            #                            # Tracking (Zeile darüber) bleibt aktiv für die
            #                            # visited_tiles-Anzeige + schnelle Reaktivierung.

        # ── FRONTIER-REWARD: maximale Distanz vom Start ──────────
        # Belohnt NUR eine neue größte Entfernung vom Startpunkt (globale
        # Weltkoordinaten). Im Kreis / zurück laufen bringt nichts → sanfter,
        # gerichteter Pull nach außen, auch über den langen Route-30-Korridor.
        # Telescoping: Summe über die Episode = finale Max-Distanz × 0.05,
        # also bewusst klein (überlagert Kämpfe/Fangen nicht).
        # NICHT mit factor multiplizieren – neues Terrain ist immer voll wert.
        cur_global = self._gct.global_with_fallback(map_group, map_number, x, y)
        dist_from_start = ((cur_global[0] - self._start_global[0]) ** 2
                           + (cur_global[1] - self._start_global[1]) ** 2) ** 0.5
        if dist_from_start > self._max_dist_from_start:
            reward += (dist_from_start - self._max_dist_from_start) * 0.06
            self._max_dist_from_start = dist_from_start

        # ── NEUE KARTE BETRETEN ───────────────────────────────
        if map_key not in self._visited_maps:
            self._visited_maps.add(map_key)
            if map_key not in self._trap_maps:
                # Echte neue Karte: einmaliger Entdeckungs-Bonus.
                reward += 23.0 * factor

        # ── ABLENKUNGSGEBÄUDE: PERMANENTE STRAFE ─────────────
        # Jeder Schritt IN einem Trap-Gebäude kostet -0.3.
        # → KI lernt: nicht reingehen, und wenn doch: sofort raus.
        # Einmal-Strafe reicht nicht – nach dem Eintritt wäre Bleiben kostenlos.
        if map_key in self._trap_maps:
            reward -= 0.3

        # ── BEISPIEL 2: POKEMON FANGEN ───────────────────────
        # Wenn die Anzahl gefangener Pokemon gestiegen ist → großer Bonus!
        # Das ist das Hauptziel: erstes Pokemon fangen.
        new_caught = caught_count - self._prev_caught

        if new_caught > 0:
            reward += 25.0 * new_caught   # +25 pro gefangenem Pokemon

        # ── BEISPIEL 3: TEAM-POKEMON ERHALTEN ────────────────
        # Wenn die Anzahl der Team-Pokemon gestiegen ist (Starter bekommen,
        # Pokemon gefangen und nicht sofort Pokedex, etc.)
        new_party = party_count - self._prev_party
        if new_party > 0:
            reward += 5.0 * new_party   # +2 pro neuem Team-Pokemon

        xp_gain = max(0, XP_count - self._prev_XP)
        if xp_gain > 0:
            reward += xp_gain*0.3*factor
            self._episode_xp += xp_gain
        
        # ── HP-HEILUNGS-REWARD ────────────────────────────────
        # Kleiner Bonus wenn HP außerhalb des Kampfes gestiegen sind
        # (Pokemon Center, Trank verwenden).
        # RATIO-basiert (nicht absolut!) damit kein Reward Hacking möglich ist:
        #   Absolut wäre: +27 Reward bei voller Heilung, aber Verlust max -1.0
        #   → netto +26 pro Heilzyklus = Exploit!
        # Ratio: max +0.5 bei voller Heilung (0% → 100% HP).
        if not in_battle:
            hp_ratio_prev_heal = self._prev_hp / self._prev_max_hp
            hp_ratio_curr_heal = current_hp / max(max_hp, 1)
            if hp_ratio_curr_heal > hp_ratio_prev_heal:
                reward += (hp_ratio_curr_heal - hp_ratio_prev_heal) * 0.5
        
        # ── KAMPFBEGINN: TRACKING INITIALISIEREN ─────────────────
        # Wichtig: _battle_min_enemy_hp IMMER auf 0 setzen, nie auf enemy_hp.
        # Grund: Nach einer Flucht bleibt der RAM-Restwert der Gegner-HP noch
        # mehrere Schritte stehen (z.B. 17/17 obwohl kein Kampf läuft).
        # Wenn dann ein neuer Kampf startet würde enemy_hp diesen Restwert lesen
        # und _battle_min_enemy_hp falsch initialisieren → falsche Damage-Rewards.
        # Lösung: Immer auf 0 → Lazy-Init setzt den echten Wert sobald er geladen ist.
        if in_battle and not self._prev_in_battle:
            self._battle_start_hp            = 0   # Lazy-Init setzt den echten Wert
            self._battle_min_enemy_hp        = 0   # Lazy-Init setzt den echten Wert
            self._enemy_defeated_this_battle = False

        # ── LAZY INIT: Kampfintro abwarten ───────────────────────
        # D116 (BATTLE_TYPE) wechselt sofort auf "Wild" beim Kampfstart,
        # aber ENEMY_HP (0xD0FF/D100) wird erst NACH der Intro-Animation geladen.
        # → battle_min_enemy_hp ist noch 0 → Schadenscheck feuert nicht.
        # Sobald erstmals gültige HP > 0 erscheinen: nachinit.
        if in_battle and self._battle_min_enemy_hp == 0 and enemy_hp > 0:
            self._battle_start_hp     = enemy_hp
            self._battle_min_enemy_hp = enemy_hp

        # ── SCHADEN-REWARD (robuste Min-HP Methode) ───────────────
        # Problem mit prev_enemy_hp: RAM-Wert aktualisiert sich nicht
        # immer innerhalb der 24 Frames → Schaden "fällt durchs Raster".
        #
        # Lösung: wir merken uns das MINIMUM der Gegner-HP im laufenden
        # Kampf. Jede neue Unterschreitung dieses Minimums = echter Schaden.
        # So wird jede HP-Änderung exakt einmal belohnt, egal wann genau
        # der RAM sich ändert.
        if in_battle and self._battle_min_enemy_hp > 0:
            if enemy_hp < self._battle_min_enemy_hp:
                damage_dealt = self._battle_min_enemy_hp - enemy_hp
                reward += damage_dealt * 0.2 * factor
                self._battle_min_enemy_hp = enemy_hp
                if enemy_hp == 0:
                    self._enemy_defeated_this_battle = True

        # ── CATCHABLE ZONE REWARD ─────────────────────────────
        # Gegner auf < 30% HP geschwächt aber noch am Leben?
        # Das ist der ideale Moment zum Pokéball werfen.
        # Kleiner Bonus pro Schritt in diesem Zustand → KI lernt "abschwächen, nicht KO-en".
        if (in_battle and enemy_hp > 0
                and self._battle_start_hp > 0
                and enemy_hp <= self._battle_start_hp * 0.3):
            reward += 0.02

        # ── DIVERSITÄTS-REWARD ────────────────────────────────
        # Neue Pokemon-ART im Team (nicht nur Anzahl) → großer Bonus.
        # Unterscheidet sich vom party-count-Reward: fängt man zweimal denselben Sentret,
        # gibt es nur beim ersten Mal diesen Bonus.
        new_species = party_species - self._prev_party_species
        if new_species:
            reward += 10.0 * len(new_species)

        # ── FLUCHT-STRAFE ─────────────────────────────────────
        # Kampf endete UND Gegner hatte noch HP UND wurde nicht besiegt
        # → KI ist geflohen.
        # _enemy_defeated_this_battle verhindert Fehlalarm wenn Gegner
        # stirbt + Kampf in denselben 24 Frames endet.
        if (self._prev_in_battle and not in_battle
                and self._prev_enemy_hp > 0
                and not self._enemy_defeated_this_battle):
            reward -= 3.0

        # ── A-TASTE IM KAMPF ──────────────────────────────────
        # DEAKTIVIERT: Hat zu Reward Hacking geführt.
        # Das Model hat gelernt A zu spammen ohne echte Kämpfe zu führen
        # (A auf Run/Item Menü statt auf Angriff) → +0.05 × 2048 = +102/Episode.
        # Der Schaden-Reward und XP-Reward reichen als Anreiz für echte Kämpfe.
        # if in_battle and self.actions[self._last_action] == 'a':
        #     reward += 0.05
        # ── BEISPIEL 4: HP-VERLUST BESTRAFEN ─────────────────
        #Wenn HP gesunken sind → kleiner negativer Reward.
        #ACHTUNG: Nicht zu groß machen, sonst kämpft die KI gar nicht mehr!
        
        hp_ratio_prev = self._prev_hp / self._prev_max_hp
        hp_ratio_curr = current_hp / max(max_hp, 1)
        hp_loss = hp_ratio_prev - hp_ratio_curr
        if hp_loss > 0:
            reward -= hp_loss * 1.0   # Max -1.0 wenn KI von voll auf 0 HP fällt
        

        # ── BEISPIEL 5: ZEITSTRAFE ───────────────────────────
        # Kleiner negativer Reward pro Schritt → KI soll effizient sein.
        # Verhindert dass die KI einfach nichts tut.
        #
        if reward==0:
            reward -= 0.002   # -0.002 pro Schritt
        
        # ACHTUNG: Mit Exploration-Reward kombinieren: die KI soll erkunden
        # ABER nicht zu lange brauchen. Teste beide Varianten!
        #
        # TODO: Aktiviere diese Zeile wenn du magst!

        # ── DEIN REWARD SYSTEM ───────────────────────────────
        # TODO: Füge hier deine eigenen Reward-Ideen ein!
        # Ideen:
        #   • Belohne bestimmte Map-Bereiche (z.B. Wildnis betreten = +0.1)
        #   • Belohne Geld verdienen
        #   • Belohne Abzeichen gewinnen (+10 pro Abzeichen)
        #   • Bestrafe Ohnmacht (party_count wird 0 = -5)

        self._reward_history.append(reward)
        return reward

    def _update_prev_state(self):
        """Aktualisiert den gespeicherten Vorherigen-Zustand."""
        self._prev_position      = read_player_position(self.pyboy)
        hp, max_hp               = read_hp(self.pyboy)
        self._prev_hp            = hp
        self._prev_max_hp        = max(max_hp, 1)
        self._prev_enemy_hp      = read_enemy_hp(self.pyboy)
        self._prev_in_battle     = read_in_battle(self.pyboy)
        self._prev_caught        = read_caught_pokemon_count(self.pyboy)
        self._prev_party         = read_party_count(self.pyboy)
        self._prev_party_species = read_party_species(self.pyboy)
        self._prev_XP            = read_total_party_xp(self.pyboy)

    def _check_terminated(self) -> bool:
        """
        Prüft ob die Episode normal enden soll (terminated=True).

        Mögliche Abbruchbedingungen:
        • KI-Pokemon alle ohnmächtig (party_count = 0)
        • Erstes Pokemon gefangen (Ziel erreicht!)
        • Bestimmter Punkt im Spiel erreicht

        Returns:
            True  = Episode beenden, neues reset() wird aufgerufen
            False = Episode läuft weiter
        """
        # Option 1: Episode endet wenn alle Pokemon ohnmächtig sind
        party_count = read_party_count(self.pyboy)
        if party_count == 0 and self._steps > 10:
            
            # Nur wenn wir wirklich Pokemon hatten (steps > 10 verhindert
            # False-Positives beim Start wo party_count noch 0 sein kann)
            return True
        
            

        if len(self._reward_history) == self._reward_history.maxlen and max(self._reward_history) < 0:
            return True

        # Option 2: Ziel erreicht! Erstes Pokemon gefangen.
        # Kommentiere das aus wenn du das als Abbruchbedingung nutzen willst.
        # if read_caught_pokemon_count(self.pyboy) > 0:
        #     return True   # Ziel erreicht – starte neue Episode!

        return False

    def _get_info(self) -> dict:
        """
        Gibt Zusatzinfos als Dictionary zurück.

        Diese werden nicht vom KI-Agenten genutzt, aber sie sind nützlich
        für TensorBoard-Logging und unser Debugging.
        """
        map_group, map_number, x, y = read_player_position(self.pyboy)
        hp, max_hp = read_hp(self.pyboy)
        return {
            "steps":           self._steps,
            "episode_reward":  self._episode_reward,
            "position":        (map_group, map_number, x, y),
            "hp_ratio":        hp / max(max_hp, 1),
            "party_count":     read_party_count(self.pyboy),
            "caught_count":    read_caught_pokemon_count(self.pyboy),
            "badge_count":     read_badge_count(self.pyboy),
            "visited_tiles":   len(self._visited_positions),
            "visited_maps":    len(self._maps_seen - self._trap_maps),
            "episode_xp":      self._episode_xp,
            "level":           read_level(self.pyboy),
            "action_history":  self._action_history if self.record_actions else [],
        }

    # ──────────────────────────────────────────────────────────
    #  SAVE-STATE VERWALTUNG
    # ──────────────────────────────────────────────────────────

    def save_current_state(self, path: str = SAVE_STATE_PATHS[0]):
        """
        Speichert den aktuellen Emulator-Zustand als Save-State-Datei.

        Nützlich um während des Spielens einen guten Startpunkt zu markieren.

        Beispiel:
            env = PokemonGoldEnv()
            env.reset()
            # ... ein paar Schritte spielen bis zum gewünschten Punkt ...
            env.save_current_state("mein_startpunkt.state")
        """
        if self.pyboy is None:
            print("⚠ PyBoy nicht initialisiert. Rufe erst reset() auf.")
            return

        state_io = io.BytesIO()
        self.pyboy.save_state(state_io)
        state_bytes = state_io.getvalue()

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(state_bytes)
        print(f"✓ Save-State gespeichert: {path}")
