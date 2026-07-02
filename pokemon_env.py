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
import os
import traceback
import faulthandler
from collections import deque
from datetime import datetime

from pyboy import PyBoy

from config import *
from ram_reader import (
    get_all_ram_features,
    read_player_position,
    read_hp,
    read_enemy_hp,
    read_party_count,
    read_party_total_hp,
    read_party_total_level,
    read_party_species,
    read_caught_pokemon_count,
    read_badge_count,
    read_exp,
    read_total_party_xp,
    read_has_egg,
    read_level,
    read_money,
    read_pp,
    read_in_battle,
    read_battle_type,
    print_game_state,
)
from global_coords import GlobalCoordinateTransform

import sys
sys.stdout.reconfigure(encoding='utf-8')


def saturation_decay(K: float, farmed: float) -> float:
    """
    Sättigungs-Faktor ∈ (0, 1] für wiederholtes Grinden auf derselben Karte.

    Gnadenzone: die ersten ~K Einheiten geben vollen Reward (Rückgabe 1.0),
    danach hyperbolischer Abfall (Halbwert bei ~2K, erreicht nie exakt 0).
    Wird gemeinsam für XP- und Schadens-Reward genutzt; "farmed" = die bereits
    auf der aktuellen Karte gefarmten XP (kanonisches Grind-Maß).

        decay = min(1, K / (1 + farmed))
    """
    return min(1.0, K / (1.0 + farmed))


# ──────────────────────────────────────────────────────────────
#  NATIVE-CRASH-DIAGNOSE (faulthandler)
# ──────────────────────────────────────────────────────────────
# Ein PyBoy/SDL-Segfault ist KEINE Python-Exception → die step()/reset()-
# Wrapper können ihn nicht fangen, der Worker stirbt UNTERHALB von Python
# (genau das aktuelle Symptom: EOFError im Hauptprozess, KEIN crash_logs/).
# faulthandler druckt bei SIGSEGV/SIGABRT den Python-Stack aller Threads
# direkt auf stderr → erscheint in der Konsole unmittelbar VOR dem EOFError
# und zeigt die exakte Zeile, an der der PyBoy-Aufruf gestorben ist.
_FAULTHANDLER_ON = False

def _enable_faulthandler_once():
    """Aktiviert faulthandler genau einmal pro Prozess (Worker wie Hauptprozess)."""
    global _FAULTHANDLER_ON
    if not _FAULTHANDLER_ON:
        try:
            faulthandler.enable(all_threads=True)
            _FAULTHANDLER_ON = True
        except Exception:
            pass


# Route in drei Legs: erst nach Westen (NBT→Cherrygrove), dann nach Norden
# (Cherrygrove→Route 31), dann in Violet City nach Nordwesten (zu Turm/Arena).
ROUTE_LEG1 = {(24, 4), (24, 3)}            # NBT, Route 29           → Fortschritt WESTEN
ROUTE_LEG2 = {(26, 3), (26, 1), (26, 2)}   # Cherrygrove, Route 30/31 → Fortschritt NORDEN

# Meilenstein-Boni: ZUSÄTZLICH zum generischen +8 beim ersten Betreten dieser
# Karten (einmalig pro Episode, via _visited_maps). Diskrete Ziele, die sich nicht
# über einen Richtungs-Skalar abbilden lassen. Turm-Etagen ANSTEIGEND → Sog nach oben.
MILESTONE_BONUS = {
    (10, 5): 30.0,    # Violet City BETRETEN (durchs Torhaus) – starker Sog über die Route-31-Schwelle
    #(3, 1):  10.0,   # Knofensa-Turm Basis betreten (Optional vor der Arena)
    #(3, 2):  10.0,   # Knofensa-Turm 1. Stock – ansteigend, zieht nach oben (statt nur rein/raus)
    #(3, 3):  10.0,   # Knofensa-Turm 2. Stock (oben) – höchste Turm-Belohnung
    (10, 7): 30.0,    # Violet-Arena (Falkner) betreten – wird mit dem KAMPF-ZUSTAND
                      # skaliert (HP/PP-Score × ARENA_READY_FLOOR-Formel, s. BEISPIEL 3):
                      # geheilt = volle 30, angeschlagen = bis runter auf 7.5
    (3 ,29): 20,      # Union Cave
    (8, 7): 20.0      # Azaelea City
}

# ══════════════════════════════════════════════════════════════════════
#  ZENTRALE REWARD-KONFIGURATION  ·  hier tunen, nicht im Code
#  Strafen sind als POSITIVE Beträge definiert (im Code via  reward -= ... ).
# ══════════════════════════════════════════════════════════════════════
# Exploration & Fortschritt
NEW_TILE_REWARD    = 0.1     # pro neuer Position (BEISPIEL 1)
PROGRESS_REWARD    = 2.0     # pro Fortschritts-Einheit entlang der Route (BEISPIEL 2)
NEW_MAP_REWARD     = 8.0     # einmalig pro neuer Nicht-Trap-Karte (BEISPIEL 3)
#   (Karten-Meilensteine Violet/Turm/Arena → MILESTONE_BONUS oben)
# Kampf & Stärker-werden
DMG_REWARD         = 0.05     # × zugefügter Schaden × Karten-Sättigung (BEISPIEL 9)
XP_REWARD          = 0.15     # × XP-Gewinn × Karten-Sättigung (BEISPIEL 7)
LEVEL_UP_REWARD    = 3.0     # pro gewonnenem Team-Level (Fänge ausgeschlossen)
TRAINER_WIN_REWARD = 7.0     # pro besiegtem Trainer
NEW_SPECIES_REWARD = 9.0     # pro neuer Pokémon-ART im Team (BEISPIEL 11)
NEW_PARTY_REWARD   = 5.0     # pro neuem Team-Mitglied (BEISPIEL 6)
CATCH_REWARD       = 12.0      # pro gefangenem Pokémon (BEISPIEL 5)
# Heilen
HEAL_REWARD        = 3.0     # HP-Auffüllung, quadratisch in der Lücke (freiwilliger PC)
PP_REFILL_REWARD   = 5.0     # PP-Auffüllung, quadratisch
# Große Meilensteine
BADGE_REWARD       = 400.0    # pro gewonnenem Orden (Arena-Sieg)
# Strafen (positive Beträge; im Code  reward -= ... )
TRAP_PENALTY       = 0.1     # pro Schritt in einem Ablenkungs-Gebäude (BEISPIEL 4)
SCROLL_PENALTY     = 0.003   # pro Schritt Bag-/Menü-Geflatter (≥50 Schritte ohne Schaden)
FLEE_PENALTY       = 3.0     # Flucht aus einem kämpfbaren Kampf (BEISPIEL 12)
HP_LOSS_PENALTY    = 2.0     # × HP-Verlust-Anteil pro Schritt (BEISPIEL 13)
TIME_PENALTY       = 0.002   # pro Schritt ganz ohne Reward (BEISPIEL 14)
# Sonderfall: Flucht aus AUSSICHTSLOSEM Kampf (≥100 Schritte ohne Schaden) → BONUS
FLEE_STUCK_BONUS   = 0.7

EGG_REWARD         = 150
PC_ENTRY_REWARD    = 30      # erstes Betreten des Violet-Pokécenters (10,10) pro Episode
                             # → zieht die KI in den Ei-„Entdeckungsraum" (NPC sitzt dort)

# ── EI-ABHOL-PHASE (Orden JA, Ei NEIN) ────────────────────────────────────
# Nach dem 1. Orden ist das nächste Spielziel das Togepi-Ei im Violet-PC
# (Route 32 ist ohne Ei ohnehin ge-gated). In dieser Phase wird die Navigation
# UMGESCHALTET: Routen-Gain pausiert, stattdessen zieht ein eigener Gain zum
# Pokécenter (egg_nav_progress, BEISPIEL 2b) + frischer PC-Meilenstein; die
# Arena wird zur Trap-Map. Mit dem Ei (Ei-LINIE: Ei ODER Togepi) endet die
# Phase DAUERHAFT → der normale Routen-Gain übernimmt wieder.
EGG_NAV_REWARD     = 2.0     # pro Kachel Annäherung an den PC (Skala wie PROGRESS_REWARD)
EGG_PC_MILESTONE   = 10.0    # erstes PC-Betreten WÄHREND der Phase (eigener Flag, s. Beispiel-Block)

# ── Arena-Meilenstein × Kampf-Zustand (Anti-„angeschlagen-in-den-Boss") ────
# Der Arena-Milestone (10,7) wird mit dem Zustand beim BETRETEN skaliert:
#   score  = 0.5·HP-Ratio (Mon 1) + 0.5·PP-Ratio   → genau das, was das PC auffüllt
#   faktor = FLOOR + (1-FLOOR)·score               → nie unter FLOOR (Sog bleibt!)
# Geheilt ankommen zahlt den vollen Milestone, angeschlagen nur den Floor-Anteil
# → belohnt die Sequenz „erst PC, dann Arena" (Win-Rate-Hebel), ohne den
# Arena-Besuch je unattraktiv zu machen. Gewichte 0.5/0.5 bei Bedarf anpassen.
ARENA_READY_FLOOR  = 0.25    # Mindest-Anteil des Arena-Milestones (0.25 → 7.5 von 30)

# Routen-Reihenfolge (Haupt-Progression) für den Backward-Reward-Fix: beim Start aus
# einem Savestate werden alle Karten VOR dem Startpunkt als "schon entdeckt" vor-
# markiert → Zurücklaufen gibt keinen +8-/Meilenstein-Bonus mehr (die wären bei einem
# echten Durchlauf längst entdeckt). Karten NICHT in der Liste (Innenräume, Seiten-
# gebiete) bleiben normal "neu entdeckbar".
ROUTE_ORDER = [
    (24, 4),   # New Bark Town
    (24, 3),   # Route 29
    (26, 3),   # Cherrygrove City
    (26, 1),   # Route 30
    (3, 70),   # Dunkelhöhle
    (26, 2),   # Route 31
    (10, 5),   # Violet City
    (10, 10),  # Violet-Pokécenter (Heilen/Savestate) – an Violet-Position
    (10, 7),   # Violet-Arena
    (10, 1),   # Route 32
    (3, 29),   # Union Cave 1F (Eingang von Route 32 Süd)
    (8, 6),    # Route 33
    (8, 7),    # Azalea City
]

# Obergrenze fürs „steps_since_map_change"-Navigationsfeature (Normierung + Sättigung):
# ab so vielen Schritten auf derselben Karte sättigt das Trödel-Signal bei 1.0.
NAV_STEPS_CAP = 300


def _savestate_group(path: str) -> str:
    """
    Ordnet einen Savestate-Pfad einer Gruppe (start/middle/end) zu – per Datei-Namen-
    Teilstring (siehe SAVESTATE_GROUPS in config.py). Robust gegen Index-Verschiebung.
    Erste passende Gruppe gewinnt; passt nichts → 'start'.
    """
    name = os.path.basename(path)
    for group, keys in SAVESTATE_GROUPS.items():
        if any(k in name for k in keys):
            return group
    return "start"


def route_progress(map_key, gx, gy):
    """
    Fortschritt ENTLANG der Route (statt radialer Distanz) – richtungs-korrekt.

      Leg 1 (NBT, Route 29):            nach Westen → 10 - gx
      Leg 2 (Cherrygrove, Route 30/31): nach Norden → 46 - gy   (ignoriert x!)
      Leg 3 (Violet City):              nach Nordwesten → 46 - gy + 0.7*(-58 - gx)

    Die Konstanten (10, 46) halten das Maß positiv UND stetig an der Naht
    Route 29 → Cherrygrove. Effekt: ab Cherrygrove zählt nur noch Norden →
    die Westküste (Sackgasse) gibt KEINEN Fortschritt mehr (vorher radial belohnt).

    Rückgabe None für (noch) nicht in der Route definierte Karten
    (Innenräume, Knofensa-Turm, Arena, Dunkelhöhle) → dort kein Richtungs-
    Fortschritt; in Violet übernehmen die Meilenstein-Boni (MILESTONE_BONUS).
    """
    # Sekundäre Achse als sanfter Nudge (Gewicht 0.5; bleibt im begehbaren Bereich
    # unter ~1.5 → keine Glitch-Verwerfung an Grenzen) gegen die bekannten Sackgassen.
    # Referenz (5.5 bzw. -42) = Koordinate der Kartengrenze → dort ist der Nudge ~0,
    # also keine Unstetigkeit am Übergang.
    if map_key == (24, 3):     # Route 29: West (Leg 1) + NORD-Nudge (weg von Süd-Sackgasse)
        return (10.0 - gx) + 0.6 * (5.5 - gy)
    if map_key in ROUTE_LEG1:  # New Bark Town
        return 10.0 - gx
    if map_key == (26, 1):     # Route 30: Nord (Leg 2) + WEST-Nudge (weg von Mr.-Pokémon-Sackgasse im Osten)
        return (46.0 - gy) + 0.6 * (-42.0 - gx)
    if map_key in ROUTE_LEG2:  # Cherrygrove, Route 31
        return 46.0 - gy
    if map_key == (10, 5):     # Violet City (Leg 3): reiner WEST-Gain, an BEIDEN Nähten stetig
        # KEIN Nord-Reward (kein gy-Term). ABER das NIVEAU muss an den Nähten passen,
        # sonst springt prog >2 → die Tod-Re-Baseline feuert fälschlich → Door-Farm
        # (Torhaus rein/raus farmt den Fortschritt jede Runde neu). Daher linear in gx
        # so kalibriert, dass prog an BEIDEN Nähten stetig anschließt (gleiche Offsets
        # wie der Env, via debug_ram gegenzuchecken – Violet-Offset ist geschätzt):
        #   Torhaus -> Route 31   global(-58,-33): muss 79.0  (= Route-31-Leg 46-gy)
        #   Naht    -> Route 32   global(-70,-27): muss 82.2  (= R32-Leg 109.2+gy)
        #   Gerade durch beide Punkte:  prog = 63.5 - 0.267*gx
        #   Probe:  gx=-58 -> 79.0     gx=-70 -> 82.2
        return 63.5 - 0.267 * gx
    if map_key == (10, 1):     # Route 32 (Leg 4): hier DREHT die Route → Gain Richtung SÜDEN
        # Route 32 liegt SÜDLICH von Violet (das Ziel NACH dem 1. Orden, Richtung Azalea).
        # Naht Violet↔R32 bei global (-70,-28): Violet-Leg3 dort = 81.2. Ab hier zählt
        # nicht mehr Norden, sondern SÜDEN (gy steigt) als Fortschritt. Die Konstante
        # 109.2 = 81.2 + 28 hält das Maß an der Naht stetig (0-Sprung); danach wächst
        # der Fortschritt monoton, je weiter die KI nach Süden läuft.
        return 109.2 + gy
    if map_key == (3, 29):     # Union Cave 1F (Warp von Route 32 Süd): Süd-Gain läuft weiter
        # WICHTIG (Dead-Zone-Fix): Route 32 reicht SÜDLICH am Höhleneingang vorbei
        # (Frontier klettert dort auf ~124). Startete die Höhle stetig bei ~121, bliebe
        # die Frontier auf 124 kleben → kein Reward trotz Süd-Lauf. Höhle startet daher
        # DEUTLICH über Route-32-Max (Eingang prog ~128 = 116+gy): der Warp-Sprung löst
        # die Teleport-Re-Baseline aus → Frontier springt ins Höhlen-Band, danach zahlt
        # Süden voll. (Warp braucht KEINE 0-Naht – die Re-Baseline übernimmt.)
        return 116.0 + gy
    if map_key == (8, 6):      # Route 33 (Leg 5): hier dreht die Route auf WESTEN
        # Höhlenausgang (-74,26) = Union-Cave-Progress 142.0 (= 116+26, seit dem
        # Dead-Zone-Fix 109.2→116). 68.0 - gx hält den Ausgang stetig (0-Sprung)
        # und wächst nach Westen. (War 61.2 = auf die ALTE Union-Konstante 109.2
        # kalibriert → 6.8-Riss am Ausgang = Door-Farm-Risiko. Nachgezogen.)
        return 68.0 - gx
    if map_key == (8, 7):      # Azalea City (Leg 6): West weiter (Eintritt von Osten)
        # Naht Route33↔Azalea bei (-80,29) → 148.0; gleiche West-Formel = 0-Sprung.
        # (61.2→68.0 zusammen mit Route 33 angehoben, sonst wandert der Riss nur
        # eine Naht weiter.) PROVISORISCH: reiner West-Gain (Eintritt von Osten).
        # Sobald die Bugsy-Arena lokalisiert ist, ggf. Nudge/Meilenstein ergänzen.
        return 68.0 - gx
    return None


# ── Geometrie der EI-ABHOL-PHASE (Konstanten siehe Reward-Block oben) ──────
EGG_NAV_MAP   = (10, 5)    # Zone: Violet City – Arena-Ausgang lokal (11,11) … PC
EGG_PC_TARGET = (17, 14)   # Ziel: PC-Tür, LOKAL in Violet City (User-Walkthrough)


def egg_nav_progress(map_key, x, y):
    """
    Fortschritts-Maß der EI-ABHOL-PHASE (Orden ja, Ei nein): Nähe zum Violet-
    Pokécenter. Nur in Violet City (EGG_NAV_MAP) definiert und in LOKALEN
    Koordinaten gerechnet – Start (Arena-Ausgang) und Ziel (PC) liegen auf
    derselben Karte, ein Offset ist unnötig.

    Skala: invertierte Manhattan-Distanz (+20), damit „näher = höher" – gleiche
    Gain-Richtung wie route_progress, gleiche Frontier-Mechanik nutzbar.
      Arena-Ausgang (11,11) → 11.0    PC-Tür (17,14) → 20.0
    Pro Geh-Schritt ändert sich der Wert um genau ±1 → bleibt unter der
    2.0-Stale-/2.1-Gain-Schwelle. None außerhalb der Zone (Innenräume, andere
    Karten) → Tracking pausiert, exakt wie beim Haupt-System.
    """
    if map_key != EGG_NAV_MAP:
        return None
    return 20.0 - (abs(x - EGG_PC_TARGET[0]) + abs(y - EGG_PC_TARGET[1]))


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

    # Pro-Karte Grind-Sättigung: K = "Gnadenzone" in XP, bis zu der voller Reward
    # gilt (decay = min(1, K/(1+farmed))); danach Abfall, Halbwert bei ~2K.
    # Gilt gemeinsam für XP- UND Schadens-Reward. Höher = mehr Farming erlaubt.
    # Nicht gelistete Karten nutzen den Default.
    GRIND_SATURATION_DEFAULT = 150
    # Werte frei tunen. Höher = mehr Kämpfe/Episode werden hier voll belohnt; niedriger
    # = drängt schneller weiter (Anti-Camping). Begrenzt nur das Reward-SIGNAL, nicht das
    # tatsächliche Leveln. Alle hier außer Route 30 stehen vorerst auf dem Default 120.
    GRIND_SATURATION_PER_MAP = {
        (26, 1): 250,   # Route 30  – höheres Gebiet, mehr Farming erlaubt
        (26, 2): 400,   # Route 31  – letzte Route vor Violet (hoch = Camping-Risiko)
        (10, 1): 800,   # Route 32  – südlich von Violet, gute Pokémon
        (3,  1): 300,   # Knofensa-Turm 1F – Sages + Wildpokémon, also grindbar
        (3, 29): 500,   # Union Cave
        (8, 6): 500,    # Route 33 - zwischen Union Cave und Azaelea
        (10, 7): 9990,   # Violet-Arena (Falkner)
        # (?, ?): 120,  # Dunkelhöhle – Map-ID noch unbekannt; (Bank,Nummer) aus debug_ram nachtragen
    }

    def __init__(self, render_mode=None, record_actions=False, verbose=True):
        """
        Initialisiert die Umgebung.

        Args:
            render_mode    : "rgb_array" zum Rendern, None für headless Training
            record_actions : True = speichert alle Aktionen für späteres Replay
            verbose        : False = keine Konsolenausgabe (für Trainings-Subprozesse)
        """
        super().__init__()

        # Native PyBoy/SDL-Crashes (Segfault) sichtbar machen: druckt bei einem
        # solchen Crash den Python-Stack auf stderr (siehe Helfer oben).
        _enable_faulthandler_once()

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

        # Robustheit: letzte gültige Beobachtung (Notfall-Fallback bei Crash)
        # und Flag, dass PyBoy beim nächsten reset() hart neu gestartet werden muss.
        self._last_obs        = None
        self._needs_hard_reset = False

        # Interner Zustand
        self._steps           = 0
        self._episode_reward  = 0.0
        self._episode_xp      = 0              # Gesammelte XP in dieser Episode
        self._xp_per_map      = {}             # XP pro Karte diese Episode (Pro-Karte-Sättigung)
        self._last_action     = -1             # Zuletzt ausgeführte Aktion (für Reward-Berechnung)
        self._action_history  = []              # Für Replay aufzeichnen (falls record_actions=True)
        self._reward_history  = deque(maxlen=N_STEPS_WITHOUT_REWARD)  # Letzten N_STEPS_WITHOUT_REWARD Rewards für Early-Stop-Check

        # Zustandsverfolgung für Reward-Berechnung
        # (was war der Zustand im LETZTEN Schritt?)
        self._prev_position   = (0, 0, 0, 0)   # (map_group, map, x, y)
        self._prev_hp         = 0              # Eigene HP (Totodile)
        self._prev_max_hp     = 1
        self._prev_egg_state  = 0
        self._prev_total_hp   = 0              # Gesamt-Team-HP letzter Schritt (Blackout-Erkennung)
        self._prev_total_level = 0             # Summe der Team-Level letzter Schritt (Level-Up-Reward)
        self._blackout_pending = False         # Team ohnmächtig → folgenden Auto-Heal NICHT belohnen
        self._pc_heals        = 0              # freiwillige Pokécenter-Heilungen diese Episode (Tracking)
        self._prev_enemy_hp              = 0     # Gegner-HP im letzten Schritt
        self._prev_in_battle             = False # Kampfstatus im letzten Schritt
        self._battle_start_hp            = 0     # Gegner-HP zu Kampfbeginn (für Catchable-Zone)
        self._battle_min_enemy_hp        = 0     # Minimum der Gegner-HP im aktuellen Kampf
        self._enemy_defeated_this_battle = False # Gegner besiegt? (verhindert falsche Flucht-Strafe)
        self._battle_steps_no_damage     = 0     # Schritte im Kampf ohne je Schaden (Stuck-Detektor)
        self._is_trainer_battle          = False # läuft gerade ein Trainerkampf? (für Sieg-Zähler)
        self._trainers_defeated          = 0     # besiegte Trainer in dieser Episode (Logging)
        self._prev_caught         = 0
        self._prev_badges         = 0              # Orden-Anzahl im letzten Schritt (Meilenstein)
        self._prev_party          = 0
        self._prev_XP             = 0
        self._prev_pp             = 0              # Gesamt-PP im letzten Schritt (für Auffüll-Reward)
        self._max_pp_seen         = 1              # höchste je gesehene Gesamt-PP (PP-Ratio-Referenz)
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

        # Frontier-Reward: Ursprung ist IMMER der Welt-Anker (0,0) = New Bark Town.
        self._start_global        = (0, 0)
        self._max_dist_from_start = 0.0     # Fortschritts-FRONTIER (Reward-Basis + Metrik)
        self._prev_prog           = None    # prog im letzten Schritt (2-Schritt-Glitch-Filter)
        self._stable_prog         = None    # letzter BESTÄTIGTER prog – großer Sprung dagegen = Teleport/Tod

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
    #  ROBUSTHEIT: Worker-Absturz abfangen (SubprocVecEnv)
    # ──────────────────────────────────────────────────────────
    # Stirbt im Worker-Subprozess eine Exception (PyBoy-Glitch, seltener RAM-
    # Lese-Edge-Case, ...), bricht die multiprocessing-Pipe → der Hauptprozess
    # crasht mit EOFError/BrokenPipeError und das GESAMTE Training (alle Envs)
    # ist weg. step()/reset() sind deshalb dünne Wrapper um die eigentliche
    # Logik (_step_impl/_reset_impl): ein Fehler wird in crash_logs/ geschrieben
    # (echte Ursache!) und die Episode sauber beendet bzw. PyBoy hart neu
    # gestartet – der Worker überlebt, die Pipe bleibt heil.

    def _zero_observation(self) -> dict:
        """Notfall-Beobachtung in der Form des observation_space (alles 0)."""
        return {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }

    def _log_worker_error(self, where: str):
        """Schreibt den vollständigen Traceback nach crash_logs/ (pro Prozess)."""
        try:
            os.makedirs("crash_logs", exist_ok=True)
            fname = os.path.join("crash_logs", f"env_crash_pid{os.getpid()}.log")
            with open(fname, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Fehler in {where} "
                        f"(Schritt {getattr(self, '_steps', '?')}):\n")
                f.write(traceback.format_exc())
        except Exception:
            pass   # Logging darf NIEMALS selbst den Worker killen

    def _hard_reset_pyboy(self):
        """PyBoy-Instanz komplett neu aufsetzen (nach hartem Glitch)."""
        try:
            if self.pyboy is not None:
                self.pyboy.stop()
        except Exception:
            pass
        self.pyboy = None
        self._init_pyboy()

    def step(self, action: int):
        """Robuster Wrapper um _step_impl – fängt jeden Worker-Crash ab."""
        try:
            obs, reward, terminated, truncated, info = self._step_impl(action)
            self._last_obs = obs
            return obs, reward, terminated, truncated, info
        except Exception:
            self._log_worker_error("step")
            # Episode abbrechen → SB3 ruft reset() auf, das PyBoy heilt.
            self._needs_hard_reset = True
            obs = self._last_obs if self._last_obs is not None else self._zero_observation()
            return obs, 0.0, True, False, {"steps": getattr(self, "_steps", 0),
                                           "crash_recovered": True}

    def reset(self, seed=None, options=None):
        """Robuster Wrapper um _reset_impl – startet PyBoy bei Bedarf hart neu."""
        for attempt in range(3):
            try:
                if self._needs_hard_reset:
                    self._hard_reset_pyboy()
                    self._needs_hard_reset = False
                obs, info = self._reset_impl(seed=seed, options=options)
                self._last_obs = obs
                return obs, info
            except Exception:
                self._log_worker_error(f"reset (Versuch {attempt + 1}/3)")
                self._needs_hard_reset = True   # nächster Versuch mit frischem PyBoy
        # Alle Versuche fehlgeschlagen: Notfall-Beobachtung, damit der Worker lebt.
        return self._zero_observation(), {"crash_recovered": True}

    # ──────────────────────────────────────────────────────────
    #  GYMNASIUM INTERFACE
    # ──────────────────────────────────────────────────────────

    def _reset_impl(self, seed=None, options=None):
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
            self._episode_group = _savestate_group(chosen_path)   # für Per-Gruppe-Logging
        else:
            # Kein State vorhanden: Intro läuft durch
            # HINWEIS: Das ist nicht ideal. Erstelle lieber einen Save-State!
            self._episode_group = "start"

        # Interne Zähler zurücksetzen
        self._steps          = 0
        self._episode_reward = 0.0
        self._episode_xp     = 0
        self._prev_egg_state = False
        self._xp_per_map     = {}
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

        # Navigations-Gedächtnis (De-Aliasing gegen Pendeln) – in _get_observation
        # berechnet. None = erster Schritt nach reset (noch keine Bewegung/kein Wechsel).
        self._prev_nav_pos    = None
        self._prev_nav_map    = None
        self._nav_steps_since = 0
        self._nav_map_changed = 0.0
        self._nav_vel         = (0.0, 0.0)

        # Frontier-Reward: Ursprung ist IMMER (0,0) = New Bark Town (Welt-Anker),
        # NICHT der Episoden-Start. So misst die Distanz echten Fortschritt Richtung
        # Ziel – egal von welchem Save-State aus gestartet wird; Zurücklaufen Richtung
        # NBT gibt nie Reward.
        # _max_dist startet bei der Distanz der Startposition zum Ursprung, damit nur
        # Fortschritt DARÜBER HINAUS zählt (sonst würde der Startpunkt selbst belohnt).
        # Innen-Start (to_global None): Basislinie erst am ersten Außen-Schritt setzen.
        self._start_global        = (0, 0)
        sb, sn, sx, sy            = self._prev_position
        start_g                   = self._gct.to_global(sb, sn, sx, sy)
        start_prog                = route_progress((sb, sn), *start_g) if start_g is not None else None
        if start_prog is not None:
            self._max_dist_from_start = start_prog   # Basislinie = Pfad-Fortschritt am Start
            self._prev_prog           = start_prog
            self._stable_prog         = start_prog
        else:
            self._max_dist_from_start = 0.0          # Innen-Start: erst draußen einschwingen
            self._prev_prog           = None
            self._stable_prog         = None

        hp, max_hp          = read_hp(self.pyboy)
        self._prev_hp       = hp
        self._prev_max_hp   = max(max_hp, 1)
        self._prev_total_hp = read_party_total_hp(self.pyboy)
        self._prev_total_level = read_party_total_level(self.pyboy)
        self._blackout_pending = False
        self._pc_heals      = 0
        self._prev_caught   = read_caught_pokemon_count(self.pyboy)
        self._prev_badges   = read_badge_count(self.pyboy)
        self._prev_party    = read_party_count(self.pyboy)
        self._prev_XP       = read_total_party_xp(self.pyboy)
        self._prev_pp       = read_pp(self.pyboy)
        self._max_pp_seen   = max(self._prev_pp, 1)
        self._battle_steps_no_damage = 0
        self._is_trainer_battle      = False
        self._trainers_defeated      = 0
        self._prev_egg_state = read_has_egg(self.pyboy)
        self._pc_egg_entered = False   # erstes Betreten des Violet-PC (10,10) diese Episode?

        # ── EI-ABHOL-PHASE: eigenes Gain-Tracking Richtung PC (BEISPIEL 2b) ──
        # 1:1-Klon der bewährten _stable_prog-Mechanik, nur mit egg_nav_progress
        # als Metrik. prev/stable None = noch keine Messung; die Basislinie setzt
        # der erste STABILE Zonen-Kontakt (ohne Reward) – wie beim Haupt-System.
        self._egg_nav_prev     = None
        self._egg_nav_stable   = None
        self._egg_nav_frontier = 0.0
        self._pc_egg_phase_entered = False   # PC-Meilenstein der Phase schon kassiert?

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
            # (26, 5) Pokemon Center EG → KEIN Trap mehr: KI soll hier heilen lernen
            #         (PP-Auffüll-Reward zieht sie hin; -0.3/Schritt würde das sabotieren)
            (20, 1),   # Pokemon Center OG – KI farmt dort den Map-Bonus ✗
            (26, 6), (26, 7), (26, 8),   # Wohnhäuser
            (26, 9),   # Haus auf Route 30 – KI läuft dort rein, kein Mehrwert
            (26, 10),  # Mr. Pokémons Haus – KI läuft dort rein, kein Mehrwert
            # Violet City Gebäude (aus User-Walkthrough) – Arena (10,7) und das
            # Pokécenter sind bewusst NICHT dabei (Turm (3,1) war es früher auch,
            # ist jetzt aber Trap – siehe unten).
            (10, 6), (10, 9), (10, 11),(10,8),   # Akademie / Markt / Wohnhaus o.ä.
            # Alph-Ruinen-Komplex bei Route 32 (aus User-Walkthrough). Hinweis:
            # Bank 3 enthält auch den Knofensa-Turm (3,1) – der ist JETZT Trap
            # (KI fand aus dem mehrstöckigen Turm nicht mehr heraus).
            (10, 12), (10, 16),                  # Torhäuser zu den Alph-Ruinen
            (3, 22), (3, 24), (3, 27), (3, 28),  # Alph-Ruinen-Innenräume
            (3, 1),                              #Knofensa Turm
            (8,3),(8,2),                         #Pokemarkt Azaelea und Haus

        }
        # EI-ABHOL-PHASE: Startet die Episode bereits MIT Orden (Gym_after_boss,
        # PC_before_egg, Route-32-States, …), ist die Arena (10,7) von Beginn an
        # Trap – nach dem Orden gibt es dort nichts mehr, der Sog muss RAUS zum
        # PC zeigen. (Gewinnt die KI den Orden erst WÄHREND der Episode, passiert
        # dasselbe live im Orden-Block von _calculate_reward.)
        if self._prev_badges >= 1:
            self._trap_maps.add((10, 7))
        self._cleared_maps  = {
            (24, 4),   # New Bark Town   ✓ verifiziert
            (24, 3),   # Route 29        ✓ verifiziert
            # (26, 5) = Pokemon Center → NICHT cleared, voller Bonus (Heilen wichtig!)
        } | self._trap_maps   # Trap-Maps sind immer auch "abgegrast"
        map_group, map_number, _, _ = self._prev_position
        self._visited_maps  = self._cleared_maps | {(map_group, map_number)}
        self._initial_map_count = len(self._visited_maps)  # (nicht mehr für Anzeige genutzt)
        self._maps_seen     = {(map_group, map_number)}    # echte Zählung: Startkarte zählt mit
        # Backward-Reward-Fix: bei einem Tiefen-Start alle Karten VOR dem Startpunkt
        # auf der Route als "besucht" vor-markieren → kein +8/Meilenstein fürs Zurück-
        # laufen. NUR _visited_maps (Reward-Gate), NICHT _maps_seen (ehrliche Zählung).
        if (map_group, map_number) in ROUTE_ORDER:
            self._visited_maps |= set(ROUTE_ORDER[:ROUTE_ORDER.index((map_group, map_number))])
        self._prev_party_species = read_party_species(self.pyboy)

        obs  = self._get_observation()
        info = self._get_info()

        return obs, info

    def _step_impl(self, action: int):
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
        # EIN nativer tick(count)-Aufruf statt FRAMES_PER_ACTION einzelner:
        # emuliert dieselben Frames (Taste durchgehend gedrückt), rendert aber
        # nur das LETZTE Bild statt jedes Frame → schneller UND drastisch weniger
        # native tick()-Aufrufe, in denen die PyBoy-"access violation" auftritt.
        # sound=False: Audio wird nie gelesen → APU-Emulation sparen (kein
        # Einfluss auf Spiellogik, RAM oder Beobachtung – nur Tempo).
        self.pyboy.tick(FRAMES_PER_ACTION, True, False)
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
            # RGBA → Graustufen in EINEM Durchgang (spart den RGB-Zwischenschritt;
            # Alpha wird in beiden Wegen ignoriert → pixelgleich zu RGBA→RGB→GRAY).
            screen_gray = cv2.cvtColor(screen_rgba, cv2.COLOR_RGBA2GRAY)
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
            "ram_features": np.concatenate([
                get_all_ram_features(self.pyboy, self._gct),   # 13 RAM-Features
                self._nav_features(),                          # +4 Navigations-Features
            ]).astype(np.float32),
        }

    def _nav_features(self) -> np.ndarray:
        """
        Navigations-Gedächtnis (4 Features) gegen Pendeln / State-Aliasing:
          • Richtung X/Y : geglättete Bewegungsrichtung (EMA). De-aliased „vorwärts"
            vs „zurück" an derselben Position → bricht das Hin-und-her.
          • map_changed  : 1.0 im Schritt eines Kartenwechsels (z.B. grad aus Gebäude raus).
          • steps_since  : wie lange schon auf dieser Karte (Trödel-/Stuck-Signal).
        Wird HIER einmal pro Schritt berechnet UND der Zustand fortgeschrieben.

        HYBRIDE Position für die Richtung: DRAUSSEN global (stetig an Seams), DRINNEN
        lokal x/y (im Raum bewegt sich x/y → Richtung funktioniert auch in Arena/Gebäude;
        global wäre dort eingefroren). Der ±1-Clamp fängt den Frame-Wechsel an der Tür ab.
        """
        mg, mn, x, y = read_player_position(self.pyboy)
        g       = self._gct.to_global(mg, mn, x, y)
        cur_pos = g if g is not None else (x, y)   # draußen global · drinnen lokal
        cur_map = (mg, mn)
        if self._prev_nav_pos is None:
            self._nav_map_changed = 0.0                          # erster Schritt nach reset
        else:
            self._nav_map_changed = 1.0 if cur_map != self._prev_nav_map else 0.0
            if self._nav_map_changed:
                self._nav_steps_since = 0
            else:
                self._nav_steps_since = min(self._nav_steps_since + 1, NAV_STEPS_CAP)
            # geglättete Geschwindigkeit (EMA); Schritt-Delta auf ±1 geklemmt
            # (Warp-/Seam-Sprünge und Frame-Wechsel an der Tür).
            dx = max(-1.0, min(1.0, cur_pos[0] - self._prev_nav_pos[0]))
            dy = max(-1.0, min(1.0, cur_pos[1] - self._prev_nav_pos[1]))
            a  = 0.3
            self._nav_vel = (a * dx + (1 - a) * self._nav_vel[0],
                             a * dy + (1 - a) * self._nav_vel[1])
        self._prev_nav_pos = cur_pos
        self._prev_nav_map = cur_map
        return np.array([
            (self._nav_vel[0] + 1.0) / 2.0,         # Richtung X [0,1] (0.5 = Stillstand)
            (self._nav_vel[1] + 1.0) / 2.0,         # Richtung Y [0,1]
            self._nav_map_changed,                  # 0/1
            self._nav_steps_since / NAV_STEPS_CAP,  # [0,1] (Trödel)
        ], dtype=np.float32)

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
        total_hp                    = read_party_total_hp(self.pyboy)
        party_count                 = read_party_count(self.pyboy)
        party_species               = read_party_species(self.pyboy)
        caught_count                = read_caught_pokemon_count(self.pyboy)
        XP_count                    = read_total_party_xp(self.pyboy)
        enemy_hp                    = read_enemy_hp(self.pyboy)
        in_battle                   = read_in_battle(self.pyboy)
        current_pp                  = read_pp(self.pyboy)
        badge_count                 = read_badge_count(self.pyboy)
        has_egg                     = read_has_egg(self.pyboy)

        map_key                     = (map_group, map_number)
        self._maps_seen.add(map_key)   # ehrliche Zählung jeder tatsächlich betretenen Karte

        # ── EI-ABHOL-PHASE? (Orden JA, Ei NEIN) ────────────────────────────
        # In der Phase wird die Navigation umgeschaltet: Routen-Gain AUS (Beispiel 2),
        # PC-Gain AN (Beispiel 2b), Arena = Trap, PC-Meilenstein scharf. has_egg ist
        # die EI-LINIE (Ei ODER Togepi) und der Orden bleibt für immer → die Phase
        # ist MONOTON (aus → an → für immer aus), kein Flip-Flop-Farm möglich.
        egg_phase = badge_count >= 1 and not has_egg

        # Karten-IDs (debug_ram.py + Disassembly verifiziert):
        #   New Bark Town (24,4), Route 29 (24,3) → cleared (nur Vorbelegung in reset)
        #   Cherrygrove (26,3), Route 30 (26,1), Route 31 (26,2) → echte neue Karten
        # Der frühere Level-Faktor (4/Level) ist ENTFERNT – das Grinding regelt jetzt
        # allein die Per-Karte-Sättigung (map_decay, siehe unten). _cleared_maps wird
        # nur noch in reset() benutzt (Vorbelegung von _visited_maps → kein +23-Bonus
        # auf bekannten Karten).

        # ── PER-KARTE GRIND-SÄTTIGUNG (für XP + Schaden gemeinsam) ────
        # Misst in XP, wie viel auf DIESER Karte schon gegrindet wurde, und dämpft
        # damit XP- UND Schadens-Reward gleichermaßen (eine Quelle der Wahrheit).
        # K pro Karte wählbar (GRIND_SATURATION_PER_MAP), Default 150.
        map_grind = self._xp_per_map.get(map_key, 0)
        map_K     = self.GRIND_SATURATION_PER_MAP.get(map_key, self.GRIND_SATURATION_DEFAULT)
        map_decay = saturation_decay(map_K, map_grind)

        # ── BEISPIEL 1: EXPLORATION REWARD (Such-Exploration) ──────────────
        # Belohne das Betreten einer NEUEN Position – treibt die KI, Fläche
        # abzudecken und so Ausgänge/Wege um Barrieren herum zu finden.
        # Pro Karte begrenzt (endliche Kacheln) → kein Dauer-Grind.
        current_pos_key = (map_group, map_number, x, y)
        if current_pos_key not in self._visited_positions:
            self._visited_positions.add(current_pos_key)
            reward += NEW_TILE_REWARD   # neue Position → Such-Exploration

        # ── BEISPIEL 2: FORTSCHRITTS-REWARD (Pfad-Progression) ─────────────
        # _max_dist_from_start = Fortschritts-FRONTIER (Reward-Basis + Metrik/Video).
        #   • Stale-Filter: prog zählt erst, wenn ZWEI Messungen in Folge nah sind (≤2).
        #   • Teleport/Tod-Erkennung über den LETZTEN STABILEN Wert (_stable_prog, jeden
        #     bestätigten Schritt aktualisiert): springt prog >2 ggü. _stable_prog, ist es
        #     Teleport/Tod/erststabil → Frontier auf die neue Position SETZEN (auch nach
        #     unten), KEIN Reward → danach zahlt Wiederhochlaufen wieder VOLL. Eine Tür/
        #     Grenze (Wiedereinstieg am ~selben Ort, Sprung ≤2) ist KEIN Teleport →
        #     Frontier bleibt → kein Hin-und-her-Farm. (Robuster als ein einmaliger
        #     Check nach der None-Lücke: fängt den Respawn auch bei Stale-Frames.)
        #   • Sonst kontinuierliche Vorwärtsbewegung (gain ≤2) → Reward, Frontier zieht mit.
        # Drinnen/ungemappt (prog None) → Tracking pausieren.
        # In der EI-ABHOL-PHASE pausiert der Routen-Gain KOMPLETT (prog=None, wie in
        # Innenräumen): der Violet-Leg zieht nach WESTEN, das PC liegt aber ÖSTLICH
        # des Arena-Ausgangs – zwei gleichzeitige, gegenläufige Gains würden sich
        # sabotieren. Beispiel 2b übernimmt; nach der Phase resumed dieses Tracking
        # sauber über Stale-Filter + Re-Baseline (Wiedereinstieg zahlt nichts).
        cur_global = self._gct.to_global(map_group, map_number, x, y)
        prog = (route_progress(map_key, *cur_global)
                if (cur_global is not None and not egg_phase) else None)
        if prog is None:
            self._prev_prog = None
        else:
            if self._prev_prog is not None and abs(prog - self._prev_prog) <= 2.0:
                # zwei nahe Messungen in Folge → prog ist VERTRAUENSWÜRDIG
                if self._stable_prog is None or abs(prog - self._stable_prog) > 2.0:
                    self._max_dist_from_start = prog     # Teleport/erststabil → Basislinie, KEIN Reward
                else:
                    gain = prog - self._max_dist_from_start
                    if 0.0 < gain <= 2.1:
                        reward += gain * PROGRESS_REWARD   # kontinuierliche Bewegung → Reward
                        self._max_dist_from_start = prog
                self._stable_prog = prog
            self._prev_prog = prog

        # ── BEISPIEL 2b: EI-ABHOL-NAVIGATION (nur in der Phase Orden-ohne-Ei) ──
        # Ersetzt in der Phase den Routen-Gain: zieht vom Arena-Ausgang (11,11) zum
        # Pokécenter (17,15), beides lokal Violet City. EXAKT dieselbe bewährte
        # Mechanik wie Beispiel 2 (Stale-Filter, >2-Re-Baseline OHNE Reward,
        # gain≤2.1-Kappe): Tod→PC-Respawn zahlt nichts (Sprung >2 → Re-Baseline),
        # PC-Tür rein/raus farmt nichts (Frontier bleibt), Zonen-Austritt pausiert.
        egg_prog = egg_nav_progress(map_key, x, y) if egg_phase else None
        if egg_prog is None:
            self._egg_nav_prev = None
        else:
            if self._egg_nav_prev is not None and abs(egg_prog - self._egg_nav_prev) <= 2.0:
                # zwei nahe Messungen in Folge → egg_prog ist VERTRAUENSWÜRDIG
                if self._egg_nav_stable is None or abs(egg_prog - self._egg_nav_stable) > 2.0:
                    self._egg_nav_frontier = egg_prog    # Teleport/erststabil → Basislinie, KEIN Reward
                else:
                    e_gain = egg_prog - self._egg_nav_frontier
                    if 0.0 < e_gain <= 2.1:
                        reward += e_gain * EGG_NAV_REWARD   # Annäherung an den PC → Reward
                        self._egg_nav_frontier = egg_prog
                self._egg_nav_stable = egg_prog
            self._egg_nav_prev = egg_prog

        # ── BEISPIEL 3: NEUE KARTE BETRETEN ───────────────────────────────
        if map_key not in self._visited_maps:
            self._visited_maps.add(map_key)
            if map_key not in self._trap_maps:
                # Echte neue Karte: einmaliger Entdeckungs-Bonus.
                reward += NEW_MAP_REWARD
                # Meilensteine (Arena/Union Cave/…): kräftiger Extra-Bonus obendrauf.
                milestone = MILESTONE_BONUS.get(map_key, 0.0)
                if map_key == (10, 7) and milestone > 0.0:
                    # Arena skaliert mit dem KAMPF-ZUSTAND beim Betreten (Konstanten-
                    # Block: ARENA_READY_FLOOR). HP = Mon 1 (der Haupt-Kämpfer);
                    # PP-Referenz = _max_pp_seen (dieselbe wie beim Heal-Reward).
                    # Greift praktisch nur den badge=0-Erstanlauf: mit Orden ist die
                    # Arena Trap bzw. via ROUTE_ORDER vor-markiert → kein Milestone.
                    hp_ratio  = current_hp / max(max_hp, 1)
                    pp_ratio  = min(1.0, current_pp / max(self._max_pp_seen, 1))
                    score     = 0.5 * hp_ratio + 0.5 * pp_ratio
                    milestone *= ARENA_READY_FLOOR + (1.0 - ARENA_READY_FLOOR) * score
                reward += milestone

        # ── BEISPIEL 4: ABLENKUNGSGEBÄUDE (PERMANENTE STRAFE) ─────────────
        # Jeder Schritt IN einem Trap-Gebäude kostet TRAP_PENALTY.
        # → KI lernt: nicht reingehen, und wenn doch: sofort raus.
        # Einmal-Strafe reicht nicht – nach dem Eintritt wäre Bleiben kostenlos.
        if map_key in self._trap_maps:
            reward -= TRAP_PENALTY

        # ── BEISPIEL 5: POKEMON FANGEN ───────────────────────
        # Wenn die Anzahl gefangener Pokemon gestiegen ist → großer Bonus!
        # Das ist das Hauptziel: erstes Pokemon fangen.
        new_caught = caught_count - self._prev_caught

        if new_caught > 0:
            reward += CATCH_REWARD * new_caught

        # ── ORDEN GEWONNEN (Arena-Sieg, größter Meilenstein) ─────────────
        # Der Badge-Counter (popcount von 0xD857) steigt nur beim Orden-Gewinn.
        # HINWEIS: 0xD857 ist die dokumentierte G/S-Johto-Orden-Adresse, aber noch
        # nicht empirisch bestätigt → beim ersten Falkner-Sieg verifizieren. Solange
        # die Adresse stimmt, feuert das nur beim echten Orden (kein Reward-Hacking).
        new_badges = badge_count - self._prev_badges
        if new_badges > 0:
            reward += BADGE_REWARD * new_badges   # 1. Orden = das eigentliche Ziel in Violet
            # Ab jetzt EI-ABHOL-PHASE: die Arena wird SOFORT zur Trap-Map – dort
            # gibt es nichts mehr, die Schritt-Strafe drückt die KI raus Richtung
            # PC. Dauerhaft (auch nach dem Ei); reset() setzt es für Orden-Starts.
            self._trap_maps.add((10, 7))
            self._cleared_maps.add((10, 7))

        # ── BEISPIEL 6: TEAM-POKEMON ERHALTEN ────────────────
        # Wenn die Anzahl der Team-Pokemon gestiegen ist (Starter bekommen,
        # Pokemon gefangen und nicht sofort Pokedex, etc.)
        new_party = party_count - self._prev_party
        if new_party > 0:
            reward += NEW_PARTY_REWARD * new_party

        # ── BEISPIEL 7: XP-GEWINN (PRO-KARTE gesättigt) ──────
        # Summe der Team-XP gestiegen → Reward, mit diminishing returns PRO KARTE:
        # je mehr XP auf DIESER Karte schon gefarmt, desto weniger pro XP.
        # → Lange auf Route 29 campen sättigt schnell (kaum noch Reward).
        #   In ein neues/höheres Gebiet ziehen setzt die Sättigung zurück
        #   (frische Karte = voller XP-Reward). Genau gewünscht: Farmen in
        #   höheren Gebieten bleibt belohnt, Festsitzen wird unattraktiv.
        # XP-Reward × map_decay (Per-Karte-Sättigung, oben einmal berechnet).
        # Kein Level-Faktor mehr – die Sättigung regelt das Grinding allein:
        #   Route 29 (K=150)  → sättigt früh, langes Campen unattraktiv
        #   Route 30 (K=1000) → höheres Gebiet, viel Farming erlaubt bevor es sättigt
        xp_gain = max(0, XP_count - self._prev_XP)
        if xp_gain > 0:
            reward += xp_gain * XP_REWARD * map_decay
            self._episode_xp += xp_gain
            self._xp_per_map[map_key] = map_grind + xp_gain

        # ── LEVEL-UP-REWARD ──────────────────────────────────────────────
        # Belohnt das Hochleveln (Summe der Team-Level steigt). Gröberes, schwer
        # farmbares Signal als reine XP. Fänge ausgeschlossen (party_count steigt
        # dann → würden sonst doppelt zählen, der Fang-Bonus deckt das schon ab).
        total_level = read_party_total_level(self.pyboy)
        if party_count == self._prev_party and total_level > self._prev_total_level:
            reward += (total_level - self._prev_total_level) * LEVEL_UP_REWARD

        # ── BEISPIEL 8: HP- & PP-HEILUNG (außerhalb Kampf) ────────────
        # Belohnt Heilen (Pokemon Center) – QUADRATISCH in der "Lücke"
        # (deficit = 1 - ratio): Heilen aus fast leerem Zustand zählt viel
        # stärker als die letzten Prozente.
        #   Potential Φ(r) = 1 - (1-r)²  →  reward = Φ_curr - Φ_prev
        #                                          = deficit_prev² - deficit_curr²
        # Telescoping (pfadunabhängig → kein Chunk-Farming). Maxima:
        #   HP volle Heilung 0→100 %  ≈ +0.8  (MUSS < HP-Verlust-Strafe 1.0 sein,
        #                                       sonst Selbstschaden→Heilen-Exploit)
        #   PP volle Auffüllung       ≈ +6.0  (kein PP-Verlust-Penalty → exploit-frei;
        #                                       quadratisch schützt vor "oft auftanken")
        # Beispiel HP: erste 10 % aus leer ≈ +0.15, letzte 10 % auf voll ≈ +0.008.
        # Blackout merken: Gesamt-Team-HP fällt auf 0 = alle ohnmächtig. Der
        # darauffolgende Auto-Heal im Pokécenter ist KEIN freiwilliger Besuch und
        # soll daher KEINEN Heil-Bonus geben (sonst Belohnung fürs Sterben).
        
        if total_hp == 0 and self._prev_total_hp > 0:
            self._blackout_pending = True

        if not in_battle:
            # PP-Maximum immer kalibrieren (unabhängig vom Heal-Reward).
            if current_pp > self._max_pp_seen:
                self._max_pp_seen = current_pp

            hp_ratio_prev_heal = self._prev_hp / self._prev_max_hp
            hp_ratio_curr_heal = current_hp / max(max_hp, 1)
            hp_up = hp_ratio_curr_heal > hp_ratio_prev_heal
            pp_up = current_pp > self._prev_pp

            # Tod-Auto-Heal vs. freiwilliger Pokécenter-Besuch:
            # Der ERSTE Heal nach einem Blackout ist der Respawn-Heal → KEIN Bonus
            # (Flag verbraucht). Jeder andere Heal-auf-mehr = freiwilliger PC-Besuch
            # → voller (quadratischer) Bonus + als pc_heal gezählt.
            if hp_up or pp_up:
                if self._blackout_pending:
                    self._blackout_pending = False        # Tod-Heal → kein Bonus
                else:
                    if hp_up:                             # HP quadratisch in der Lücke
                        d_prev = 1.0 - hp_ratio_prev_heal
                        d_curr = 1.0 - hp_ratio_curr_heal
                        reward += (d_prev ** 2 - d_curr ** 2) * HEAL_REWARD
                    if pp_up:                             # PP quadratisch
                        pp_d_prev = 1.0 - self._prev_pp / self._max_pp_seen
                        pp_d_curr = 1.0 - current_pp  / self._max_pp_seen
                        reward += (pp_d_prev ** 2 - pp_d_curr ** 2) * PP_REFILL_REWARD
                    self._pc_heals += 1                   # freiwilliger PC-Heal (Tracking)

            # Heilen IM Kampf
            # if  in_battle:
            #     # HP (echte Ratio aus RAM)
            #     hp_ratio_prev_heal = self._prev_hp / self._prev_max_hp
            #     hp_ratio_curr_heal = current_hp / max(max_hp, 1)
            #     if hp_ratio_curr_heal > hp_ratio_prev_heal:
            #         d_prev = 1.0 - hp_ratio_prev_heal
            #         d_curr = 1.0 - hp_ratio_curr_heal
            #         reward += (d_prev ** 2 - d_curr ** 2) * 0.1


        # ── SETUP A: KAMPFBEGINN-TRACKING (kein Reward) ─────────────────
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
            self._battle_steps_no_damage     = 0   # Stuck-Zähler für neuen Kampf zurücksetzen
            self._is_trainer_battle          = (read_battle_type(self.pyboy) == 2)  # 2 = Trainer

        # ── SETUP B: LAZY INIT – Kampfintro abwarten (kein Reward) ───────────────────────
        # D116 (BATTLE_TYPE) wechselt sofort auf "Wild" beim Kampfstart,
        # aber ENEMY_HP (0xD0FF/D100) wird erst NACH der Intro-Animation geladen.
        # → battle_min_enemy_hp ist noch 0 → Schadenscheck feuert nicht.
        # Sobald erstmals gültige HP > 0 erscheinen: nachinit.
        if in_battle and self._battle_min_enemy_hp == 0 and enemy_hp > 0:
            self._battle_start_hp     = enemy_hp
            self._battle_min_enemy_hp = enemy_hp

        # ── BEISPIEL 9: SCHADEN-REWARD (Min-HP) + KEIN-SCHADEN-ZÄHLER ─────
        # Min-HP-Methode: wir merken das MINIMUM der Gegner-HP im Kampf; jede neue
        # Unterschreitung = echter Schaden (robust gegen verzögerte RAM-Updates).
        #
        # Im SELBEN Block läuft der Kein-Schaden-Zähler, damit beide konsistent sind:
        #   • Schaden gemacht → Zähler auf 0  (Kampf läuft, kein Geflatter).
        #   • kein Schaden    → Zähler +1.
        # Er misst also AUFEINANDERFOLGENDE schadenlose Schritte (resettet bei JEDEM
        # Treffer) – erkennt Festfahren/Bag-Geflatter MITTEN im Kampf, nicht nur am
        # Anfang. Speist die Geflatter-Strafe (unten) UND den Flucht-Bonus (≥50).
        if in_battle and self._battle_min_enemy_hp > 0:
            if enemy_hp < self._battle_min_enemy_hp:
                damage_dealt = self._battle_min_enemy_hp - enemy_hp
                reward += damage_dealt * DMG_REWARD * map_decay   # gleiche Per-Karte-Sättigung wie XP
                self._battle_min_enemy_hp    = enemy_hp
                self._battle_steps_no_damage = 0            # Treffer → Zähler zurück
                if enemy_hp == 0:
                    self._enemy_defeated_this_battle = True
            else:
                self._battle_steps_no_damage += 1           # kein Schaden diesen Schritt

        # Ei eingesammelt

        if has_egg and not self._prev_egg_state:   # nur 0→1 (Ei GEHOLT), NICHT Schlüpfen 1→0
            reward += EGG_REWARD

        # Erstes Betreten des Violet-Pokécenters (10,10) pro Episode → zieht die KI in den
        # „Entdeckungsraum" fürs Ei (der NPC sitzt dort). Eigener Flag statt Milestone, weil
        # (10,10) via ROUTE_ORDER für Gym-/Route32-Starts vor-markiert wäre → Milestone
        # würde dort gar nicht feuern.
        if map_key == (10, 10) and not self._pc_egg_entered:
            self._pc_egg_entered = True
            reward += PC_ENTRY_REWARD

        # PC-Meilenstein der EI-ABHOL-PHASE (+EGG_PC_MILESTONE): feuert auch, wenn
        # das PC diese Episode schon VOR dem Orden besucht wurde (heilen) – der
        # Anreiz „geh JETZT (wieder) zum PC" muss nach dem Orden FRISCH gesetzt
        # werden. Eigener Flag statt MILESTONE_BONUS (Grund wie oben: ROUTE_ORDER
        # würde (10,10) für Gym-/Route-32-Starts vor-markieren → nie feuern).
        if egg_phase and map_key == (10, 10) and not self._pc_egg_phase_entered:
            self._pc_egg_phase_entered = True
            reward += EGG_PC_MILESTONE
        # ── BAG-/MENÜ-GEFLATTER BREMSEN ──────────────────────────────────
        # Ab 50 AUFEINANDERFOLGENDEN Schritten ohne Schaden -0.003/Schritt → zielloses
        # Scrollen im Beutel (z.B. nach leeren Pokébällen) wird teuer; die KI soll
        # angreifen oder fliehen. Sobald ein Treffer fällt, ist der Zähler 0 → in
        # echten Kämpfen feuert das nicht. Flucht-Bonus (≥75) bleibt erhalten.
        if in_battle and self._battle_steps_no_damage >= 50:
            reward -= SCROLL_PENALTY

        # ── BEISPIEL 10: CATCHABLE ZONE (DEAKTIVIERT) ───────────────
        # AUS: +0.02/Schritt bei geschwächtem Gegner belohnte das VERWEILEN.
        # Bei leeren PP konnte die KI nicht beenden → hielt den Gegner am Leben und
        # kassierte weiter → verlängerter Struggle. Der Stuck-Detektor greift hier
        # nicht (er zählt nur Kämpfe OHNE jeden Schaden; hier wurde Schaden gemacht).
        # Fang-Reward (+25) bleibt als Anreiz; das Abschwächen lernt die KI darüber.
        # if (in_battle and enemy_hp > 0
        #         and self._battle_start_hp > 0
        #         and enemy_hp <= self._battle_start_hp * 0.3):
        #     reward += 0.02

        # ── BEISPIEL 11: DIVERSITÄTS-REWARD (neue Art) ────────────────────────────────
        # Neue Pokemon-ART im Team (nicht nur Anzahl) → großer Bonus.
        # Unterscheidet sich vom party-count-Reward: fängt man zweimal denselben Sentret,
        # gibt es nur beim ersten Mal diesen Bonus.
        new_species = party_species - self._prev_party_species
        if new_species:
            reward += NEW_SPECIES_REWARD * len(new_species)

        # ── BEISPIEL 12: FLUCHT-STRAFE ─────────────────────────────────────
        # Kampf endete UND Gegner hatte noch HP UND wurde nicht besiegt
        # → KI ist geflohen.
        # _enemy_defeated_this_battle verhindert Fehlalarm wenn Gegner
        # stirbt + Kampf in denselben 24 Frames endet.
        if (self._prev_in_battle and not in_battle
                and self._prev_enemy_hp > 0
                and not self._enemy_defeated_this_battle):
            if self._battle_steps_no_damage >= 100:
                # Festgefahren (75+ AUFEINANDERFOLGENDE Schritte ohne Schaden, z.B.
                # leere Angriffs-PP): Fliehen ist hier RICHTIG → Bonus statt Strafe.
                reward += FLEE_STUCK_BONUS
            else:
                # Flucht aus einem kämpfbaren Kampf → weiterhin Strafe.
                reward -= FLEE_PENALTY

        # ── TRAINER BESIEGT  ────────────
        # Vor Trainern kann man nicht fliehen → ein endender Trainerkampf ist ein
        # SIEG, außer man ist ohnmächtig geworden (Blackout). battle_type==2 ist
        # verifiziert (Log zeigt "Train"). _blackout_pending (Gesamt-Team-HP→0) ist
        # robuster als "current_hp>0": ein Sieg, bei dem nur das Leit-Mon umfiel,
        # zählt jetzt korrekt mit.
        if self._prev_in_battle and not in_battle:
            if self._is_trainer_battle and not self._blackout_pending:
                self._trainers_defeated += 1
                reward += TRAINER_WIN_REWARD
            self._is_trainer_battle = False

        # ── A-TASTE IM KAMPF (DEAKTIVIERT) ──────────────────────────────────
        # DEAKTIVIERT: Hat zu Reward Hacking geführt.
        # Das Model hat gelernt A zu spammen ohne echte Kämpfe zu führen
        # (A auf Run/Item Menü statt auf Angriff) → +0.05 × 2048 = +102/Episode.
        # Der Schaden-Reward und XP-Reward reichen als Anreiz für echte Kämpfe.
        # if in_battle and self.actions[self._last_action] == 'a':
        #     reward += 0.05
        # ── BEISPIEL 13: HP-VERLUST BESTRAFEN ─────────────────
        #Wenn HP gesunken sind → kleiner negativer Reward.
        #ACHTUNG: Nicht zu groß machen, sonst kämpft die KI gar nicht mehr!
        
        hp_ratio_prev = self._prev_hp / self._prev_max_hp
        hp_ratio_curr = current_hp / max(max_hp, 1)
        hp_loss = hp_ratio_prev - hp_ratio_curr
        if hp_loss > 0:
            reward -= hp_loss * HP_LOSS_PENALTY   # voll → 0 HP kostet HP_LOSS_PENALTY
        

        # ── BEISPIEL 14: ZEITSTRAFE ───────────────────────────
        # Kleiner negativer Reward pro Schritt → KI soll effizient sein.
        # Verhindert dass die KI einfach nichts tut.
        #
        if reward==0:
            reward -= TIME_PENALTY   # kleine Zeitstrafe pro Schritt ohne Reward
        
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
        self._prev_total_hp      = read_party_total_hp(self.pyboy)
        self._prev_total_level   = read_party_total_level(self.pyboy)
        self._prev_enemy_hp      = read_enemy_hp(self.pyboy)
        self._prev_in_battle     = read_in_battle(self.pyboy)
        self._prev_caught        = read_caught_pokemon_count(self.pyboy)
        self._prev_badges        = read_badge_count(self.pyboy)
        self._prev_party         = read_party_count(self.pyboy)
        self._prev_party_species = read_party_species(self.pyboy)
        self._prev_XP            = read_total_party_xp(self.pyboy)
        self._prev_pp            = read_pp(self.pyboy)
        self._prev_egg_state     =read_has_egg(self.pyboy)

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
        
        # if read_badge_count(self.pyboy)>=1:
        #     return True

        
            

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
            "trainers_defeated": self._trainers_defeated,
            "pc_heals":        self._pc_heals,
            "violet_visited":  1 if (10, 5) in self._maps_seen else 0,
            "azaelea_visited":  1 if (8, 7) in self._maps_seen else 0,            
            "tower_visited":   1 if (3, 1)  in self._maps_seen else 0,
            "arena_visited":   1 if (10, 7) in self._maps_seen else 0,
            "union_cave_visited": 1 if (3, 29) in self._maps_seen else 0,
            "has_egg":         read_has_egg(self.pyboy),
            "savestate_group": self._episode_group,
            "visited_tiles":   len(self._visited_positions),
            "visited_maps":    len(self._maps_seen - self._trap_maps),
            "max_dist_from_start": self._max_dist_from_start,
            "episode_xp":      self._episode_xp,
            "level":           read_level(self.pyboy),
            "party_level":     read_party_total_level(self.pyboy),
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
