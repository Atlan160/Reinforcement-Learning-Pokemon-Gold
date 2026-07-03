"""
╔══════════════════════════════════════════════════════════════╗
║    POKEMON GOLD RL  ·  Zentrale Konfiguration (config.py)   ║
╚══════════════════════════════════════════════════════════════╝

Alle Parameter des Projekts befinden sich hier.
Ändere Werte in dieser Datei, um zu experimentieren –
du musst keinen anderen Code anfassen!
"""

import os

# ──────────────────────────────────────────────────────────────
#  ROM UND SPIELPARAMETER
# ──────────────────────────────────────────────────────────────

# Pfad zur Pokemon Gold ROM (.gbc Datei)
# Passe diesen Pfad an deinen Dateinamen an!
ROM_PATH = "PGV.gbc"

# Pfad zum Save-State (wird beim ersten Start automatisch erstellt)
# Ein Save-State ist ein "Snapshot" des Spielzustands – wie ein Quicksave.
# Wir starten immer vom gleichen Punkt (z.B. direkt nach dem Intro).
# HINWEIS: Erstelle diesen manuell mit PyBoy oder lasse ihn beim ersten
# Start vom Code generieren (nach einer definierten Anzahl von Frames).
# Mehrere Save-States für Curriculum Learning:
# Jede Episode wird zufällig einer dieser States geladen.
# → KI lernt beide Bereiche gleichzeitig und vergisst nichts.
# Einfach weitere States hier ergänzen wenn neue Gebiete dazukommen!
SAVE_STATE_PATHS = [
    "./Savestates/PGV.state",              # New Bark Town → Route 29 S
    "./Savestates/PGV.state",              # New Bark Town → Route 29 S
    "./Savestates/PGV.state",              # New Bark Town → Route 29 S
    "./Savestates/PGV.state",              # New Bark Town → Route 29 S
    "./Savestates/PGV.state",              # New Bark Town → Route 29 S
    "./Savestates/PGV_Route30_1.state",
    "./Savestates/PGV_Route30_2.state",
    "./Savestates/PGV_Route30_3.state",
    "./Savestates/PGV_Route30_4.state",
    "./Savestates/PGV_Route29_2.state",
    "./Savestates/PGV_Route29_3.state",
    "./Savestates/PGV_Route29_4.state",
    "./Savestates/PGV_Pokecenter_am_Schalter.state",
    "./Savestates/PGV_Pokecenter_Violet_city.state", 
    "./Savestates/PGV_Pokecenter_Violet_city_before_door.state",
    "./Savestates/PGV_Route31_1.state", # S
    "./Savestates/PGV_Route31_2.state",
    "./Savestates/PGV_Route31_3.state",
    "./Savestates/PGV_Route31_4.state", 
    "./Savestates/PGV_Violet_city.state", #M
    #"./Savestates/PGV_Knofensaturm_1.state",
    #"./Savestates/PGV_Knofensaturm_2.state",
    #"./Savestates/PGV_Knofensaturm_Boss.state",
    "./Savestates/PGV_Gym1.state", #M
    "./Savestates/PGV_Gym_boss.state", #M
    #"./Savestates/PGV_Gym_after_boss.state", #M
    "./Savestates/PGV_PC_before_egg.state", #M
    "./Savestates/PGV_after_boss_before_egg.state", #M
    "./Savestates/PGV_Pokecenter_at_assistant.state", #M
    #"./Savestates/PGV_before_egg_gate.state",  #M
    "./Savestates/PGV_Route32_2.state", # E after egg on route 32
    "./Savestates/PGV_Route32_4.state", #E
    "./Savestates/PGV_Route32_5.state", #E
    "./Savestates/PGV_Union_cave_1.state", #E
    "./Savestates/PGV_Union_cave_2.state", #E
]

# ── Savestate-Gruppen fürs GETRENNTE TensorBoard-Logging (start / middle / end) ──
# Zuordnung per Datei-NAMEN-Teilstring (NICHT Index!) → robust dagegen, dass das
# Auskommentieren/Ergänzen von Savestates die Indizes verschiebt.
# Pro Episode wird der geladene State der ERSTEN Gruppe zugeordnet, deren Baustein
# im Dateinamen vorkommt; passt nichts → "start". Trag hier einfach Namens-Bausteine ein.
# Ergebnis in TensorBoard: eigene Sektionen  start/…  middle/…  end/…
SAVESTATE_GROUPS = {
    "start":  ["PGV.state", "Route29", "Route30", "Route31","Pokecenter_am_Schalter"],                 # weit vom Ziel
    "middle": ["Gym", "Pokecenter_Violet_city", "Knofensaturm", "before_egg",
               "assistant", "Violet"],                                        # Gym-/Ei-Abhol-Bereich
    "end":    ["Route32", "Union", "Route33", "Azalea"],                      # post-Egg / Höhle weiter
}

# Wie viele GameBoy-Frames soll die KI pro Schritt sehen?
# 1 Frame = ~1/60 Sekunde Spielzeit
# 24 = ~0.4 Sekunden pro Schritt (empfohlen für Pokemon – Bewegung braucht Zeit)
FRAMES_PER_ACTION = 24

# Maximale Schritte pro Episode (Episode = ein Trainings-Durchlauf)
# 2048 Schritte × 24 Frames = ~819 Sekunden Spielzeit pro Episode
MAX_STEPS_PER_EPISODE = 4096*10    # 12288 Schritte – mehr Zeit um neue Route zu finden
N_STEPS_WITHOUT_REWARD= 500       # Etwas mehr Geduld bei Strecken ohne Reward

# Spielgeschwindigkeit beim Training (0 = so schnell wie möglich!)
# Beim Training nie auf 1 setzen – das wäre viel zu langsam.
EMULATION_SPEED = 0

# ──────────────────────────────────────────────────────────────
#  BEOBACHTUNGSRAUM (was sieht die KI?)
# ──────────────────────────────────────────────────────────────

# Größe des Bildschirm-Inputs für das CNN
# GameBoy Color Originalauflösung: 160 × 144 Pixel
# Wir skalieren auf 84 × 84 (Standard für Atari-RL, gut erforscht)
SCREEN_HEIGHT = 84
SCREEN_WIDTH  = 84

# Graustufenbild (1 Kanal) oder Farbbild (3 Kanäle)?
# Graustufen sind schneller zu trainieren und oft ausreichend.
# True  → 1 Kanal  (Graustufen)
# False → 3 Kanäle (RGB)
GRAYSCALE = True

# Anzahl der RAM-Features, die zusätzlich zum Bild eingegeben werden.
# Wird automatisch aus ram_reader.py gezählt – hier nur als Orientierung.
# Aktuell: 17 Features (lokale X/Y, globale Welt-X/Y, is_indoor, HP, Team-Größe,
#          Team-Gesamtlevel, PP, Gefangen, Badges, Ei, Kampfstatus,
#          + Navigation: Richtung X/Y, map_changed, steps_since_map)
N_RAM_FEATURES = 17

# ──────────────────────────────────────────────────────────────
#  AKTIONSRAUM (was kann die KI tun?)
# ──────────────────────────────────────────────────────────────

# Welche GameBoy-Buttons stehen zur Verfügung?
# Weniger Aktionen → schnelleres Lernen, aber weniger Flexibilität.
#
# Vollständige Liste: ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select']
#
# Für das erste Ziel (Pokemon fangen, Bewegung lernen) reichen:
# Richtungstasten + A (für Dialoge/Kämpfe bestätigen) + B (abbrechen)
# + Start (für Menü) (nicht gebraucht hier da alle relevanten Infos per Observation Variables übermittelt werden)

ACTIONS = ['up', 'down', 'left', 'right', 'a', 'b']

# ──────────────────────────────────────────────────────────────
#  PPO HYPERPARAMETER
# ──────────────────────────────────────────────────────────────
"""
PPO (Proximal Policy Optimization) – kurze Erklärung:
─────────────────────────────────────────────────────
Für Pokemon verwenden wir eine CnnPolicy (statt MlpPolicy wie bei Pong).
Das CNN verarbeitet das Spielbild, ähnlich wie ein Gehirn visuelle Infos.

Wichtige Hyperparameter für Pokemon (längere Episoden als Pong!):

  LEARNING_RATE : Schrittgröße beim Lernen
                  → 2.5e-4 ist ein guter Startwert für CNN-Policies
  N_STEPS       : Frames pro Umgebung vor jedem PPO-Update
                  → 512 bedeutet: sammle 512 Schritte, dann lerne daraus
  BATCH_SIZE    : Wie viele Samples pro Mini-Batch beim Netz-Update
  GAMMA         : Discount-Faktor (wie weit schaut die KI in die Zukunft?)
                  → 0.99 = KI denkt ~100 Schritte voraus
  ENT_COEF      : Entropie-Bonus (fördert Ausprobieren / Exploration)
                  → Pokemon braucht viel Exploration → höherer Wert nötig!
"""

LEARNING_RATE  = 1E-4
N_STEPS        = 4096       # Schritte pro Umgebung vor Update
# BATCH_SIZE: GPUs rechnen effizienter mit größeren Batches.
# RTX 4070 hat 12 GB VRAM → 512 oder 1024 sind unproblematisch.
# Größerer Batch = stabilere Gradienten, aber weniger Updates pro Datenmenge.
BATCH_SIZE     = 1024       # GPU-optimiert (statt 256 für CPU)
N_EPOCHS       = 4         # Epochen pro PPO-Update (weniger als Pong – stabiler)
GAMMA          = 0.997      # Discount-Faktor (etwas erhöht da langfrisitige Ziele erreicht werden sollen)
GAE_LAMBDA     = 0.95      # GAE-Lambda (Bias/Varianz-Tradeoff)
CLIP_RANGE     = 0.2       # PPO Clipping-Epsilon
ENT_COEF       = 0.02      # Entropie-Koeffizient – erhöht auf 0.05 um neue Gebiete zu finden
                           # (war 0.02 – höherer Wert = mehr Zufälligkeit = mehr Exploration)
VF_COEF        = 0.5       # Value-Function-Loss-Gewicht

# ──────────────────────────────────────────────────────────────
#  TRAININGS-PARAMETER
# ──────────────────────────────────────────────────────────────

# Anzahl paralleler Umgebungen
# Jede Umgebung ist eine eigene PyBoy-Instanz im eigenen Prozess.
# Der i9 14. Gen hat 24 Kerne (8P + 16E) → viel Spielraum!
# RAM-Verbrauch: ~250 MB pro Env → 16 Envs ≈ 4 GB RAM (unproblematisch)
# Faustregel: N_ENVS = halbe Kernanzahl (Betriebssystem braucht auch Kerne)
N_ENVS = 20

# Gesamte Trainingsschritte
# Mit RTX 4070 + 16 Envs läuft das Training deutlich schneller:
# ~1 Mio Schritte ≈ 20-40 Min (statt 1-2 Std auf CPU-only)
# Pokemon braucht 5-20 Mio für brauchbare Ergebnisse → starte mit 5 Mio
TOTAL_TIMESTEPS = 50_000_000

# Wie oft soll der Agent evaluiert werden (in Schritten)?
EVAL_FREQ       = 500_000_000 #damit deaktiviert

# Wie oft soll ein Checkpoint gespeichert werden?
CHECKPOINT_FREQ = 200_000

# ──────────────────────────────────────────────────────────────
#  PFADE
# ──────────────────────────────────────────────────────────────

MODEL_SAVE_PATH  = "models/pokemon_ppo_final"
LOG_PATH         = "./tensorboard_logs/"
CHECKPOINT_PATH  = "./checkpoints/"
VIDEO_PATH       = "./videos/"
