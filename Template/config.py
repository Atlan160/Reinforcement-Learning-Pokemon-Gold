"""
╔══════════════════════════════════════════════════════════════╗
║         PONG RL  ·  Zentrale Konfiguration (config.py)      ║
╚══════════════════════════════════════════════════════════════╝

Alle Parameter des Projekts befinden sich hier.
Ändere Werte in dieser Datei, um zu experimentieren –
du musst keinen anderen Code anfassen!
"""

# ──────────────────────────────────────────────────────────────
#  SPIELPARAMETER
# ──────────────────────────────────────────────────────────────

WINDOW_WIDTH   = 800    # Fensterbreite  [px]
WINDOW_HEIGHT  = 600    # Fensterhöhe    [px]
FPS            = 60     # Frames/Sekunde (nur beim Spielen mit Pygame)

PADDLE_WIDTH       = 20    # Schlägerbreite         [px]
PADDLE_HEIGHT      = 80    # Schlägerhöhe           [px]
PADDLE_SPEED       = 5     # Bewegung pro Frame     [px]
PADDLE_MAX_TILT    = 20    # Maximaler Kippwinkel   [Grad, ±]
PADDLE_TILT_SPEED  = 2     # Kippgeschwindigkeit    [Grad/Frame]

WALL_HEIGHT      = 160    # Wandhöhe           [px]
WALL_WIDTH      = 20    # Wandhöhe           [px]
WALL_SPEED       = 2     # Wand Bewegung pro Frame     [px]

BALL_SIZE          = 10    # Balldurchmesser        [px]
BALL_INITIAL_SPEED = 5.0   # Startgeschwindigkeit   [px/Frame]
BALL_MAX_SPEED_FACTOR= 3   # wie viele male schneller darf der Ball werden

# X-Positionen der Schläger (fest)
PLAYER_PADDLE_X = 30
AGENT_PADDLE_X  = WINDOW_WIDTH - 30

# ──────────────────────────────────────────────────────────────
#  FEATURE-FLAGS  →  Hier experimentieren!
# ──────────────────────────────────────────────────────────────

# Können Schläger gekippt werden?
# True  → Interessantere Physik!  Agent muss lernen Winkel zu nutzen.
# False → Klassisches Pong ohne Kippung
TILTING_ENABLED = True

# Schwierigkeit des Bot-Gegners während des Trainings (0.0 – 1.0)
# 0.7 → Bot ist berechenbar beatable → Agent bekommt positive Belohnungen
# 1.0 → Bot nahezu perfekt → Agent lernt kaum (kein positives Signal)
# 0.5 → Bot sehr schwach → Agent lernt schnell, aber gegen echte Gegner schwach
BOT_DIFFICULTY = 1.1

# ──────────────────────────────────────────────────────────────
#  PPO HYPERPARAMETER
# ──────────────────────────────────────────────────────────────
"""
Was ist PPO? (Proximal Policy Optimization, Schulman et al. 2017)
─────────────────────────────────────────────────────────────────
PPO ist ein moderner Policy-Gradient-Algorithmus.
"Policy" = Strategie = was der Agent in jedem Zustand tut.

Kern-Idee von PPO:
  Verbessere die Strategie schrittweise, aber verhindere
  zu große Änderungen auf einmal (→ "Proximal" = nah dran).

Das wird durch Clipping erreicht:
  L_clip = E[ min( r·A,  clip(r, 1−ε, 1+ε)·A ) ]
  
  wobei:
    r = π_neu(a|s) / π_alt(a|s)  ← Verhältnis neue/alte Strategie
    A = Advantage (Vorteil einer Aktion gegenüber Durchschnitt)
    ε = clip_range (0.2 bedeutet: max. ±20% Änderung)

Parameter-Erklärungen:
  LEARNING_RATE : Lernrate des neuronalen Netzes
                  → zu groß: instabil | zu klein: langsam
  N_STEPS       : Schritte die vor jedem Update gesammelt werden
                  → mehr = stabiler aber langsamer
  BATCH_SIZE    : Mini-Batch-Größe beim Netz-Training
  N_EPOCHS      : Wie oft wird über die gesammelten Daten gelernt?
                  PPO kann mehrmals über dieselben Daten → Effizienz!
  GAMMA         : Discount-Faktor γ ∈ [0,1]
                  → γ=0: nur aktuelle Belohnung zählt (kurzsichtig)
                  → γ=1: alle zukünftigen Belohnungen gleich gewichtet
                  → γ=0.99: Standard – schwach abzinsen
  GAE_LAMBDA    : λ für Generalized Advantage Estimation
                  → Balanciert Bias/Varianz beim Advantage-Schätzen
  CLIP_RANGE    : ε – maximale Policy-Änderung pro Update
  ENT_COEF      : Entropie-Bonus → fördert Exploration
                  (Agent probiert auch schlechte Aktionen aus, um zu lernen)
"""

LEARNING_RATE  = 3e-4
N_STEPS        = 512     # Schritte pro Umgebung vor Update
BATCH_SIZE     = 256      # Mini-Batch-Größe
N_EPOCHS       = 10       # Epochen pro PPO-Update
GAMMA          = 0.99     # Discount-Faktor
GAE_LAMBDA     = 0.95     # GAE-Lambda
CLIP_RANGE     = 0.2      # PPO Clipping-Parameter ε
ENT_COEF       = 0.007    # Entropie-Koeffizient

# Netzwerkarchitektur: Liste von versteckten Schichtgrößen
# [256, 256] = 2 Schichten mit je 256 Neuronen (Actor UND Critic teilen diese)
NET_ARCH = [256, 256]

# ──────────────────────────────────────────────────────────────
#  TRAININGS-PARAMETER
# ──────────────────────────────────────────────────────────────

N_ENVS           = 16          # Parallele Trainings-Umgebungen
TOTAL_TIMESTEPS  = 300*N_ENVS*N_STEPS   # Gesamte Trainingsschritte
                                 # ~10 Min auf moderner CPU
EVAL_FREQ        = 20_000       # Alle X Schritte evaluieren
CHECKPOINT_FREQ  = 200_000      # Alle X Schritte speichern

# ──────────────────────────────────────────────────────────────
#  PFADE
# ──────────────────────────────────────────────────────────────

MODEL_SAVE_PATH  = "models/pong_ppo_final"
LOG_PATH         = "./tensorboard_logs/"
CHECKPOINT_PATH  = "./checkpoints/"
