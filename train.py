"""
╔══════════════════════════════════════════════════════════════╗
║     POKEMON GOLD RL  ·  Training mit PPO (train.py)         ║
╚══════════════════════════════════════════════════════════════╝

AUSFÜHREN:
  python train.py

TRAINING BEOBACHTEN (in einem zweiten Terminal):
  tensorboard --logdir ./tensorboard_logs
  → Browser: http://localhost:6006

WIE UNTERSCHEIDET SICH DAS TRAINING VOM PONG-PROJEKT?
───────────────────────────────────────────────────────
Pong nutzte MlpPolicy (einfaches neuronales Netz für 7 Zahlen).
Pokemon nutzt MultiInputPolicy (CNN + MLP kombiniert):

  ┌────────────────────────────────────────────────────────┐
  │           MultiInputPolicy – Architektur               │
  │                                                        │
  │  screen (84×84×1)          ram_features (10 Zahlen)   │
  │       │                           │                    │
  │  ┌────▼─────┐             ┌───────▼──────┐            │
  │  │   CNN    │             │     MLP      │            │
  │  │ 3 Conv-  │             │  (kleines    │            │
  │  │ Schichten│             │   Netz)      │            │
  │  └────┬─────┘             └───────┬──────┘            │
  │       │                           │                    │
  │       └──────────┬────────────────┘                   │
  │                  ▼                                     │
  │          ┌───────────────┐                            │
  │          │  Kombinierter │                            │
  │          │  Feature-Vec. │                            │
  │          └───────┬───────┘                            │
  │                  │                                     │
  │         ┌────────┴────────┐                           │
  │         ▼                 ▼                           │
  │    ┌─────────┐     ┌─────────────┐                   │
  │    │  ACTOR  │     │   CRITIC    │                   │
  │    │ (Policy)│     │ (Value Fn.) │                   │
  │    └────┬────┘     └──────┬──────┘                   │
  │         │                 │                           │
  │    Aktion wählen    Zustand bewerten                  │
  └────────────────────────────────────────────────────────┘

CNN (Convolutional Neural Network):
  Lernt Muster im Bild zu erkennen:
  - Layer 1: Ecken und Kanten
  - Layer 2: Texturen und Formen
  - Layer 3: Komplexe Objekte (Pokemon-Sprite, Menü, etc.)

ACHTUNG – TRAINING BRAUCHT ZEIT!
  Pokemon ist viel komplexer als Pong.
  Erste sinnvolle Ergebnisse: ~500.000 – 1.000.000 Schritte
  Vollständiges Ergebnis: 5.000.000 – 20.000.000 Schritte
  Das sind Stunden bis Tage Trainingszeit!
"""

import os
import warnings
import logging
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

# TensorFlow / TensorBoard Spam unterdrücken
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
logging.getLogger("absl").setLevel(logging.ERROR)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)

from config import *
from pokemon_env import PokemonGoldEnv


# ──────────────────────────────────────────────────────────────
#  BENUTZERDEFINIERTE CALLBACKS
# ──────────────────────────────────────────────────────────────

class PokemonTrainingLogger(BaseCallback):
    """
    Callback: Wird bei jedem Schritt aufgerufen.
    Loggt Pokemon-spezifische Metriken zu TensorBoard:
      • Durchschnittlicher Reward
      • Besuchte Felder (Explorations-Fortschritt)
      • Gefangene Pokemon
      • HP-Verhältnis

    Ein Callback ist wie ein "Beobachter" der das Training überwacht
    ohne es zu beeinflussen.
    """

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq           = log_freq
        self._last_log_step     = 0
        self.episode_rewards    = []
        self.episode_lengths    = []
        self.caught_counts      = []
        self.visited_tiles      = []
        self.visited_maps       = []
        self.episode_xp         = []
        self._prev_steps        = {}
        self._prev_ep_reward    = {}
        self._prev_ep_tiles     = {}   # visited_tiles des letzten Schritts pro Env
        self._prev_ep_maps      = {}   # visited_maps des letzten Schritts pro Env
        self._prev_ep_caught    = {}   # caught_count des letzten Schritts pro Env
        self._prev_ep_xp        = {}   # episode_xp des letzten Schritts pro Env

    def _on_step(self) -> bool:
        # Episodenenden erkennen (gleiche Methode wie im Pong-Projekt)
        for i, info in enumerate(self.locals.get("infos", [])):
            curr_steps  = info.get("steps", 0)
            curr_reward = info.get("episode_reward", 0.0)
            prev_steps  = self._prev_steps.get(i, 0)

            if curr_steps < prev_steps and prev_steps > 10:
                # Episode beendet → Werte vom LETZTEN Schritt der alten Episode verwenden,
                # nicht vom aktuellen (der gehört bereits zur neuen Episode nach reset())
                self.episode_rewards.append(self._prev_ep_reward.get(i, 0.0))
                self.episode_lengths.append(prev_steps)
                self.visited_tiles.append(self._prev_ep_tiles.get(i, 0))
                self.visited_maps.append(self._prev_ep_maps.get(i, 0))
                self.caught_counts.append(self._prev_ep_caught.get(i, 0))
                self.episode_xp.append(self._prev_ep_xp.get(i, 0))

            self._prev_steps[i]      = curr_steps
            self._prev_ep_reward[i]  = curr_reward
            self._prev_ep_tiles[i]   = info.get("visited_tiles", 0)
            self._prev_ep_maps[i]    = info.get("visited_maps",  0)
            self._prev_ep_caught[i]  = info.get("caught_count",  0)
            self._prev_ep_xp[i]      = info.get("episode_xp",    0)

        # Regelmäßig in TensorBoard loggen
        steps_since_log = self.num_timesteps - self._last_log_step
        if steps_since_log >= self.log_freq and len(self.episode_rewards) > 0:
            self._last_log_step = self.num_timesteps

            n           = min(50, len(self.episode_rewards))
            mean_r      = np.mean(self.episode_rewards[-n:])
            mean_len    = np.mean(self.episode_lengths[-n:])
            mean_tiles  = np.mean(self.visited_tiles[-n:])  if self.visited_tiles  else 0
            mean_maps   = np.mean(self.visited_maps[-n:])   if self.visited_maps   else 0
            mean_caught = np.mean(self.caught_counts[-n:])  if self.caught_counts  else 0
            mean_xp     = np.mean(self.episode_xp[-n:])     if self.episode_xp     else 0
            progress    = self.num_timesteps / TOTAL_TIMESTEPS * 100

            # In TensorBoard loggen (erscheint als Kurven im Browser)
            self.logger.record("pokemon/mean_reward",        mean_r)
            self.logger.record("pokemon/mean_ep_length",     mean_len)
            self.logger.record("pokemon/mean_visited_tiles", mean_tiles)
            self.logger.record("pokemon/mean_visited_maps",  mean_maps)
            self.logger.record("pokemon/mean_caught",        mean_caught)
            self.logger.record("pokemon/mean_episode_xp",   mean_xp)
            self.logger.dump(self.num_timesteps)

            # Fortschrittsbalken in der Konsole
            bar_len = 30
            filled  = int(bar_len * progress / 100)
            bar     = "█" * filled + "░" * (bar_len - filled)

            print(
                 f"\r[{bar}] {progress:5.1f}%  "
                f"Schritte: {self.num_timesteps:>8,}  |  "
                f"Ø Belohnung: {mean_r:+.3f}  |  "
                f"Ø Karten: {mean_maps:.1f}  |  "
                f"Ø Felder: {mean_tiles:.0f}  |  "
                f"Ø Gefangen: {mean_caught:.2f}  |  "
                f"Ø XP: {mean_xp:.0f}",
                flush=True
            )

        return True


# ──────────────────────────────────────────────────────────────
#  HAUPT-TRAININGSFUNKTION
# ──────────────────────────────────────────────────────────────

def run_training(checkpoint_path: str = None):
    """
    Startet das Training – entweder frisch oder von einem Checkpoint.

    Der einzige Unterschied zwischen "neu" und "fortsetzen":
      checkpoint_path = None  → neues PPO-Modell erstellen
      checkpoint_path = "..." → Modell aus Datei laden + Umgebung verbinden

    Alles andere (Envs, Callbacks, model.learn) ist identisch.

    Args:
        checkpoint_path : Pfad zu einer .zip-Checkpoint-Datei, oder None.
    """
    is_resuming = checkpoint_path is not None

    print("\n" + "═" * 65)
    print("  POKEMON GOLD RL – TRAINING MIT PPO")
    print("═" * 65)
    print(f"  Modus:             {'Fortsetzen von ' + checkpoint_path if is_resuming else 'Frisches Training'}")
    print(f"  Algorithmus:       PPO (Proximal Policy Optimization)")
    print(f"  Policy:            MultiInputPolicy (CNN + RAM-Features)")
    print(f"  Parallele Envs:    {N_ENVS}")
    print(f"  Trainingsschritte: {TOTAL_TIMESTEPS:,}")
    print(f"  Frames/Aktion:     {FRAMES_PER_ACTION}")
    print(f"  ROM:               {ROM_PATH}")
    print("═" * 65)

    for path in ("models", LOG_PATH, CHECKPOINT_PATH):
        os.makedirs(path, exist_ok=True)

    # ── UMGEBUNGEN ERSTELLEN ──────────────────────────────────
    # SubprocVecEnv startet jede Umgebung in einem EIGENEN PROZESS.
    # Das ist ideal für den i9 14. Gen mit 24 Kernen:
    #   → 16 Envs laufen echt parallel auf 16 CPU-Kernen
    #
    # WICHTIGER WINDOWS-HINWEIS:
    #   SubprocVecEnv nutzt Python multiprocessing mit "spawn".
    #   Jeder Kindprozess importiert dieses Skript neu.
    #   Deshalb MUSS der Aufruf in if __name__ == "__main__" stehen!
    #
    # Fallback: DummyVecEnv auskommentieren falls Probleme auftreten.
    print(f"\n→ Erstelle {N_ENVS} parallele Trainingsumgebungen (SubprocVecEnv)...")

    def make_train_env():
        """
        Factory-Funktion für eine einzelne Umgebung.
        Muss eine benannte Funktion sein (kein Lambda) – multiprocessing
        kann nur pickling-bare Objekte in Kindprozesse übergeben.
        """
        env = PokemonGoldEnv(render_mode=None, verbose=False)
        return Monitor(env)

    train_env = make_vec_env(
        env_id=make_train_env,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,   # ← echte CPU-Parallelität
        # vec_env_cls=DummyVecEnv,   # ← Fallback bei Multiprocessing-Problemen
        seed=42,
    )
    eval_env = Monitor(PokemonGoldEnv(render_mode=None))

    # ── MODELL ERSTELLEN ODER LADEN ───────────────────────────
    # Was läuft auf der GPU (RTX 4070)?
    #   → CNN-Forward-Pass und PPO-Backpropagation
    # Was läuft auf der CPU (i9)?
    #   → PyBoy-Simulation aller 16 Envs parallel
    #   → Der typische RL-Bottleneck ist die Env-Simulation, nicht das Netz.
    if is_resuming:
        print(f"\n→ Lade Checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, device="cuda")
        model.set_env(train_env)   # Neue Envs verbinden (alte sind weg)
    else:
        print("\n→ Erstelle neuen PPO-Agenten mit MultiInputPolicy (CNN + MLP)...")
        # "MultiInputPolicy" versteht Dict-Beobachtungsräume.
        # Stable-Baselines3 erstellt automatisch:
        #   • Ein CNN für den "screen"-Input (3 Conv-Schichten, SB3-Standard)
        #   • Ein MLP für den "ram_features"-Input
        #   • Einen gemeinsamen Feature-Extractor der beides kombiniert
        model = PPO(
            policy          = "MultiInputPolicy",
            env             = train_env,
            learning_rate   = LEARNING_RATE,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            gamma           = GAMMA,
            gae_lambda      = GAE_LAMBDA,
            clip_range      = CLIP_RANGE,
            ent_coef        = ENT_COEF,
            vf_coef         = VF_COEF,
            device          = "cuda",   # RTX 4070
            verbose         = 0,
            tensorboard_log = LOG_PATH,
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"   Netz-Parameter: {n_params:,}")
    print(f"   Aktionsraum:    Discrete({train_env.action_space.n}) – {ACTIONS}")

    # ── CALLBACKS EINRICHTEN ─────────────────────────────────
    print("\n→ Callbacks einrichten...")

    logger_cb = PokemonTrainingLogger(log_freq=10_000)

    eval_cb = EvalCallback(
        eval_env             = eval_env,
        best_model_save_path = "./models/",
        log_path             = LOG_PATH,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = 1,   # Pokemon-Episoden sind lang → wenige reichen
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = CHECKPOINT_PATH,
        name_prefix = "pokemon_ppo",
        verbose     = 1,
    )

    # ── TRAINING STARTEN ─────────────────────────────────────
    print(f"\n→ Training läuft... ({TOTAL_TIMESTEPS:,} Schritte)")
    print("  Ctrl+C um zu unterbrechen – Modell wird trotzdem gespeichert.\n")
    print(f"  TensorBoard: tensorboard --logdir {LOG_PATH}")
    print()

    try:
        model.learn(
            total_timesteps     = TOTAL_TIMESTEPS,
            callback            = [logger_cb, eval_cb, checkpoint_cb],
            reset_num_timesteps = True,
            
        )
        print("\n\n✓ Training erfolgreich abgeschlossen!")

    except KeyboardInterrupt:
        print("\n\n⚠ Training manuell unterbrochen.")

    model.save(MODEL_SAVE_PATH)
    print(f"✓ Finales Modell gespeichert:       {MODEL_SAVE_PATH}.zip")
    print(f"✓ Bestes Modell (nach Evaluation):  ./models/best_model.zip")

    print("\n" + "─" * 65)
    print("  NÄCHSTE SCHRITTE:")
    print("  1. KI-Verhalten aufnehmen:  python record.py")
    print(f"  2. Training analysieren:    tensorboard --logdir {LOG_PATH}")
    print("─" * 65 + "\n")

    train_env.close()
    eval_env.close()


def find_latest_checkpoint() -> str | None:
    """Sucht den neuesten Checkpoint im Checkpoint-Ordner."""
    if not os.path.isdir(CHECKPOINT_PATH):
        return None
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_PATH) if f.endswith(".zip")],
        key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_PATH, f)),
        reverse=True,
    )
    if checkpoints:
        return os.path.join(CHECKPOINT_PATH, checkpoints[0])
    return None


# ──────────────────────────────────────────────────────────────
#  EINSTIEGSPUNKT
# ──────────────────────────────────────────────────────────────

# True  → neuesten Checkpoint suchen und Training fortsetzen
# False → immer frisch starten (ignoriert vorhandene Checkpoints)
RESUME_FROM_CHECKPOINT = True
 
if __name__ == "__main__":
    # WICHTIG: Dieser Guard ist für SubprocVecEnv auf Windows zwingend!
    # Ohne ihn würde jeder Kindprozess erneut Kindprozesse spawnen → Absturz.
    
    if RESUME_FROM_CHECKPOINT:
        latest = find_latest_checkpoint()
        if latest:
            print(f"→ Neuester Checkpoint gefunden: {latest}")
        else:
            print("→ Kein Checkpoint gefunden, starte frisches Training.")
        run_training(checkpoint_path="models/Iteration3.zip")   # None = frisch, Pfad = fortsetzen
    else:
        run_training()
 