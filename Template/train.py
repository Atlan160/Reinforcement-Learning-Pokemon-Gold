"""
╔══════════════════════════════════════════════════════════════╗
║         PONG RL  ·  Training mit PPO (train.py)             ║
╚══════════════════════════════════════════════════════════════╝

AUSFÜHREN:
  python train.py

TRAINING BEOBACHTEN (in einem zweiten Terminal):
  tensorboard --logdir ./tensorboard_logs
  → Browser: http://localhost:6006

WIE FUNKTIONIERT PPO?
──────────────────────
PPO = Proximal Policy Optimization (Schulman et al., 2017)
Ein Policy-Gradient-Verfahren mit Actor-Critic-Architektur.

1) ACTOR (= Policy π):
   Ein neuronales Netz, das für jeden Zustand s eine
   Wahrscheinlichkeitsverteilung über Aktionen ausgibt:
     π(a | s) = P(Aktion a | Zustand s)

2) CRITIC (= Value Function V):
   Ein neuronales Netz, das schätzt wie viel Gesamtbelohnung
   der Agent ab Zustand s noch erwarten kann:
     V(s) = E[R_t + γ·R_{t+1} + γ²·R_{t+2} + ...]

3) ADVANTAGE A(s, a):
   Wie viel BESSER ist Aktion a im Vergleich zum Durchschnitt?
     A(s, a) = Q(s, a) − V(s)
   PPO schätzt A mit Generalized Advantage Estimation (GAE).

4) PPO-LOSS (das Herzstück):
   L = −E[ min( r·A,  clip(r, 1−ε, 1+ε)·A ) ]

   wobei:
     r = π_neu(a|s) / π_alt(a|s)   ← Wie stark hat sich die Policy geändert?
     A = Advantage                  ← War diese Aktion gut oder schlecht?
     ε = clip_range (z.B. 0.2)     ← Maximale Änderung pro Update

   Das Clipping verhindert zu große Sprünge → stabiles Training!

5) PARALLELE UMGEBUNGEN:
   N_ENVS Umgebungen laufen gleichzeitig und sammeln unabhängige
   Erfahrungen. Das ist wichtig, weil:
     • Aufeinanderfolgende Frames in EINER Umgebung sind stark korreliert
     • Korrelation destabilisiert das Netz-Training
     • Parallele Envs liefern unabhängige, diverse Erfahrungen

   Trainingsdaten pro Update-Schritt: N_STEPS × N_ENVS Übergänge

   ┌── Env 1 ──┐   s₁, a₁, r₁
   ├── Env 2 ──┤   s₂, a₂, r₂
   ├── Env 3 ──┤   s₃, a₃, r₃   ──▶  PPO Update
   ├──  ...  ──┤         ...
   └── Env N ──┘   sₙ, aₙ, rₙ
"""

import os
import warnings
import logging
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')


# TensorFlow / TensorBoard Spam killen
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# pygame warning ausblenden
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pygame.pkgdata"
)

# absl logging reduzieren
logging.getLogger("absl").setLevel(logging.ERROR)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)

from config import *
from pong_env import PongEnv


# ──────────────────────────────────────────────────────────────
#  BENUTZERDEFINIERTER CALLBACK
# ──────────────────────────────────────────────────────────────

class WinRateEvalCallback(EvalCallback):
    """
    Erweitert EvalCallback um eine saubere Siegraten-Auswertung.
    Zählt Siege/Niederlagen über n_eval_episodes deterministische Spiele.
    """
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Nach jeder Evaluation die Siegrate berechnen
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if len(self.evaluations_results) > 0:
                last_rewards = self.evaluations_results[-1]
                wins   = sum(1 for r in last_rewards if r > 0)
                losses = len(last_rewards) - wins
                win_rate = wins / len(last_rewards)
                
                self.logger.record("eval/eval_win_rate", win_rate)
                self.logger.record("eval/eval_wins",     wins)
                self.logger.record("eval/eval_losses",   losses)
                self.logger.dump(self.num_timesteps)
                
                print(f"\n  Eval Siegrate: {win_rate*100:.1f}%  "
                      f"({wins}W / {losses}L aus {len(last_rewards)} Spielen)")
        
        return result
    

class TrainingLogger(BaseCallback):
    """
    Callback: Wird während des Trainings aufgerufen.
    Erkennt Episodenenden über den steps-Zähler im Info-Dict,
    da Monitor's 'episode'-Key mit DummyVecEnv nicht zuverlässig ankommt.
    """

    def __init__(self, log_freq: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq          = log_freq
        self.episode_rewards   = []
        self.episode_lengths   = []
        self._last_log_step    = 0
        self.agent_wins        = 0
        self.bot_wins          = 0
        self._prev_steps       = {}   # letzter steps-Wert pro Env
        self._prev_ep_reward   = {}   # letzter episode_reward pro Env

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            curr_steps  = info.get("steps", 0)
            curr_reward = info.get("episode_reward", 0.0)
            prev_steps  = self._prev_steps.get(i, 0)

            # Episodenende erkannt: steps-Zähler ist zurückgesprungen
            if curr_steps < prev_steps and prev_steps > 10:
                final_reward = self._prev_ep_reward.get(i, 0.0)
                self.episode_rewards.append(final_reward)
                self.episode_lengths.append(prev_steps)
                if final_reward > 0:
                    self.agent_wins += 1
                else:
                    self.bot_wins += 1

            self._prev_steps[i]     = curr_steps
            self._prev_ep_reward[i] = curr_reward

        steps_since_log = self.num_timesteps - self._last_log_step
        if steps_since_log >= self.log_freq and len(self.episode_rewards) > 0:
            self._last_log_step = self.num_timesteps

            n        = min(100, len(self.episode_rewards))
            mean_r   = np.mean(self.episode_rewards[-n:])
            mean_len = np.mean(self.episode_lengths[-n:])
            progress = self.num_timesteps / TOTAL_TIMESTEPS * 100
            total    = self.agent_wins + self.bot_wins
            win_rate = self.agent_wins / total if total > 0 else 0

            self.logger.record("custom/win_rate",       win_rate)
            self.logger.record("custom/mean_reward",    mean_r)
            self.logger.record("custom/mean_ep_length", mean_len)
            self.logger.dump(self.num_timesteps)

            bar_len = 30
            filled  = int(bar_len * progress / 100)
            bar     = "█" * filled + "░" * (bar_len - filled)

            print(
                f"\r[{bar}] {progress:5.1f}%  "
                f"Schritte: {self.num_timesteps:>8,}  |  "
                f"Ø Belohnung: {mean_r:+.3f}  |  "
                f"Ø Epis.-Länge: {mean_len:.0f}  |  "
                f"Siegrate: {win_rate*100:.1f}%  "
                f"({self.agent_wins}W / {self.bot_wins}L)",
                flush=True
            )

        return True

# ──────────────────────────────────────────────────────────────
#  HAUPT-TRAININGSFUNKTION
# ──────────────────────────────────────────────────────────────

def train():
    print("\n" + "═" * 65)
    print("  PONG RL – TRAINING MIT PPO")
    print("═" * 65)
    print(f"  Algorithmus:          PPO (Proximal Policy Optimization)")
    print(f"  Parallele Envs:       {N_ENVS}")
    print(f"  Trainingsschritte:    {TOTAL_TIMESTEPS:,}")
    print(f"  Daten pro Update:     {N_STEPS * N_ENVS:,}  ({N_STEPS} × {N_ENVS})")
    print(f"  Netzwerkarchitektur:  MLP {NET_ARCH}")
    print(f"  Kippfunktion:         {'AN ✓' if TILTING_ENABLED else 'AUS'}")
    print(f"  Bot-Schwierigkeit:    {BOT_DIFFICULTY}")
    print("═" * 65)

    # Ordner erstellen
    for path in ("models", LOG_PATH, CHECKPOINT_PATH):
        os.makedirs(path, exist_ok=True)

    # ── PARALLELE TRAININGSUMGEBUNGEN ──────────────────────────
    # make_vec_env erstellt N_ENVS parallele Kopien der Umgebung.
    # Monitor-Wrapper zeichnet Episodenstatistiken auf (r, l, t).
    # Diese werden von unserem Callback und TensorBoard genutzt.
    print(f"\n→ Erstelle {N_ENVS} parallele Trainingsumgebungen...")

    def make_train_env():
        """Factory-Funktion: erstellt eine einzelne Umgebung."""
        env = PongEnv(render_mode=None, tilting_enabled=TILTING_ENABLED)
        return Monitor(env)

    train_env = make_vec_env(
        env_id=make_train_env,
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        seed=42,
    )

    # Separate Evaluierungs-Umgebung
    # (getrennt vom Training, damit Eval-Ergebnisse unvoreingenommen sind)
    eval_env = Monitor(
        PongEnv(render_mode=None, tilting_enabled=TILTING_ENABLED)
    )

    # ── PPO-AGENT ERSTELLEN ────────────────────────────────────
    # "MlpPolicy" = Multi-Layer Perceptron Policy
    # Das neuronale Netz hat:
    #   • Input-Schicht:    obs_dim Neuronen (7 oder 9)
    #   • Versteckte Schichten: NET_ARCH = [256, 256]
    #   • Actor-Kopf:       n_actions Neuronen (Softmax → Wahrscheinlichkeiten)
    #   • Critic-Kopf:      1 Neuron (Wert des Zustands)
    #
    # Actor und Critic TEILEN die versteckten Schichten (shared backbone).
    # Das ist effizienter und in der Praxis oft besser.
    print("→ Erstelle PPO-Agenten mit MLP-Policy...")

    model = PPO(
        policy         = "MlpPolicy",
        env            = train_env,
        learning_rate  = LEARNING_RATE,
        n_steps        = N_STEPS,
        batch_size     = BATCH_SIZE,
        n_epochs       = N_EPOCHS,
        gamma          = GAMMA,
        gae_lambda     = GAE_LAMBDA,
        clip_range     = CLIP_RANGE,
        ent_coef       = ENT_COEF,
        device         ="cpu",
        policy_kwargs  = dict(net_arch=NET_ARCH),
        verbose        = 0,           # Wir nutzen unseren eigenen Logger
        tensorboard_log= LOG_PATH,
    )

    # Parameteranzahl ausgeben (Orientierung für Netzgröße)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"   Netz-Parameter: {n_params:,}")
    print(f"   Obs-Dimension:  {train_env.observation_space.shape[0]}")
    print(f"   Aktionsraum:    Discrete({train_env.action_space.n})")

    # ── CALLBACKS EINRICHTEN ──────────────────────────────────
    print("→ Callbacks einrichten...")

    # 1. Unser eigener Fortschritts-Logger
    logger_cb = TrainingLogger(log_freq=20_000)

    # 2. Eval-Callback: Evaluiert alle EVAL_FREQ Schritte und
    #    speichert das beste Modell (nach mittlerer Episodenbelohnung)
    eval_cb = WinRateEvalCallback(
        eval_env             = eval_env,
        best_model_save_path = "./models/",
        log_path             = LOG_PATH,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = 50,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    # 3. Checkpoint-Callback: Speichert den Agenten regelmäßig.
    #    Falls das Training abstürzt, kannst du von einem Checkpoint weitermachen.
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = CHECKPOINT_PATH,
        name_prefix = "pong_ppo",
        verbose     = 0,
    )

    # ── TRAINING STARTEN ──────────────────────────────────────
    print(f"\n→ Training läuft... ({TOTAL_TIMESTEPS:,} Schritte)")
    print("  Ctrl+C um zu unterbrechen – Modell wird trotzdem gespeichert.\n")
    print(f"  TensorBoard: tensorboard --logdir {LOG_PATH}")
    print()

    try:
        model.learn(
            total_timesteps      = TOTAL_TIMESTEPS,
            callback             = [logger_cb, eval_cb, checkpoint_cb],
            reset_num_timesteps  = True,
        )
        print("\n\n✓ Training erfolgreich abgeschlossen!")

    except KeyboardInterrupt:
        print("\n\n⚠ Training manuell unterbrochen.")

    # ── MODELL SPEICHERN ──────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    print(f"✓ Finales Modell gespeichert:       {MODEL_SAVE_PATH}.zip")
    print(f"✓ Bestes Modell (nach Evaluation):  ./models/best_model.zip")

    print("\n" + "─" * 65)
    print("  NÄCHSTE SCHRITTE:")
    print("  1. Gegen Agent spielen:  python play.py")
    print(f"  2. Training analysieren: tensorboard --logdir {LOG_PATH}")
    print("─" * 65 + "\n")

    train_env.close()
    eval_env.close()


# ──────────────────────────────────────────────────────────────
#  TRAINING FORTSETZEN (Checkpoint laden)
# ──────────────────────────────────────────────────────────────

def continue_training(checkpoint_path: str):
    print(f"→ Lade Checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")

    def make_train_env():
        env = PongEnv(render_mode=None, tilting_enabled=TILTING_ENABLED)
        return Monitor(env)

    train_env = make_vec_env(
        env_id=make_train_env,
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        seed=0,
    )
    eval_env = Monitor(PongEnv(render_mode=None, tilting_enabled=TILTING_ENABLED))
    model.set_env(train_env)

    # Callbacks hinzufügen – ohne diese sieht man nichts!
    logger_cb     = TrainingLogger(log_freq=20_000)
    eval_cb       = WinRateEvalCallback(
        eval_env             = eval_env,
        best_model_save_path = "./models/",
        log_path             = LOG_PATH,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = 50,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = CHECKPOINT_PATH,
        name_prefix = "pong_ppo_continued",
        verbose     = 0,
    )

    print(f"→ Setze Training fort für weitere {TOTAL_TIMESTEPS:,} Schritte...")
    try:
        model.learn(
            total_timesteps     = TOTAL_TIMESTEPS,
            callback            = [logger_cb, eval_cb, checkpoint_cb],
            reset_num_timesteps = True,  # False = Schrittzähler WEITERFÜHREN
        )
        print("\n✓ Fertig!")
    except KeyboardInterrupt:
        print("\n⚠ Unterbrochen.")

    model.save(MODEL_SAVE_PATH + "_continued")
    train_env.close()
    eval_env.close()

# ──────────────────────────────────────────────────────────────
#  EINSTIEGSPUNKT
# ──────────────────────────────────────────────────────────────

Commence_training=True
path="models/average_model.zip"

if __name__ == "__main__":


    if Commence_training:
        # python train.py checkpoints/pong_ppo_200000_steps.zip
        try:
            continue_training(path)
        except:
            print("File "+path+" not found")
            train()
            
    else:
        train()
