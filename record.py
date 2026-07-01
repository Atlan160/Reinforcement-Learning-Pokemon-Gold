"""
╔══════════════════════════════════════════════════════════════╗
║  POKEMON GOLD RL  ·  Video-Aufnahme & Replay (record.py)    ║
╚══════════════════════════════════════════════════════════════╝

AUSFÜHREN (nach dem Training):
  python record.py

WAS MACHT DIESES SKRIPT?
──────────────────────────
1. Lädt ein trainiertes Modell
2. Lässt die KI Pokemon spielen
3. Zeichnet die Aktionen auf
4. Speichert das Gameplay als Video (.mp4)

ZWEI MODI:
  MODUS 1: DIREKTE AUFNAHME
    → PyBoy läuft, KI spielt, jeden Frame als Bild speichern
    → Am Ende: Bilder zu Video zusammenfügen
    → Einfach, zuverlässig

  MODUS 2: ACTION REPLAY
    → KI spielt, nur Aktionen werden gespeichert (z.B. [0, 4, 2, 1, ...])
    → Danach: Spiel nochmal starten, Aktionen der Reihe nach ausführen,
              dabei Video aufnehmen
    → Nützlich um jederzeit einen Run nachzuspielen

  Standardmäßig nutzen wir Modus 1 (einfacher).
  Modus 2 ist als Alternative am Ende implementiert.

ABHÄNGIGKEITEN:
  pip install imageio imageio-ffmpeg
  (für Video-Export; ist in requirements.txt bereits drin)

TIPP – SAVE STATE ERSTELLEN:
  Dieses Skript kann auch einen Save-State erstellen!
  Starte das Spiel, spiele manuell bis zum Startpunkt,
  dann drücke 's' im Fenster oder nutze create_save_state().
"""

import os
import sys
import io
import numpy as np
import cv2               # Bildverarbeitung
import imageio           # Video-Export
from stable_baselines3 import PPO

from pyboy import PyBoy

from config import *
from pokemon_env import PokemonGoldEnv
from ram_reader import print_game_state, read_caught_pokemon_count, read_party_count

sys.stdout.reconfigure(encoding='utf-8')


# ──────────────────────────────────────────────────────────────
#  KONFIGURATION FÜR DIE AUFNAHME
# ──────────────────────────────────────────────────────────────

# Wie viele Episoden aufnehmen?
N_RECORD_EPISODES = 1

# Maximale Schritte pro aufgenommener Episode
MAX_RECORD_STEPS = MAX_STEPS_PER_EPISODE

# Video-Auflösung (GameBoy original: 160×144, skaliert für bessere Sichtbarkeit)
VIDEO_SCALE  = 2     # 160×4 = 640 × 144×4 = 576 Pixel
# GameBoy läuft mit ~60 FPS. Wir nehmen alle 24 Frames pro Schritt auf.
# VIDEO_SPEED = Wiedergabe-Tempo: 1 = Echtzeit, 2 = doppelt so schnell (halb so lang).
# Mehr fps = gleiche Frames in kürzerer Zeit → schnelleres Video.
VIDEO_SPEED  = 4
VIDEO_FPS    = 60 * VIDEO_SPEED   # 120 fps = 2× GameBoy-Geschwindigkeit

# Info-Overlay ein- oder ausschalten
# True  → Schritt, Reward, HP etc. werden in jedem Frame eingeblendet
# False → reines Gameplay-Video ohne Text
SHOW_OVERLAY = True

"""
SAVE_STATE_PATHS = [
    "./Savestates/PGV.state",              # New Bark Town → Route 29
    "./Savestates/PGV.state",              # New Bark Town → Route 29
    "./Savestates/PGV.state",              # New Bark Town → Route 29
    "./Savestates/PGV.state",              # New Bark Town → Route 29
    "./Savestates/PGV_Route30_1.state",
    "./Savestates/PGV_Route30_2.state",
    "./Savestates/PGV_Route30_3.state",
    "./Savestates/PGV_Route30_4.state",
    "./Savestates/PGV_Route29_2.state",
    "./Savestates/PGV_Route29_3.state",
    "./Savestates/PGV_Route29_4.state",
    "./Savestates/PGV_Pokecenter_am_Schalter.state",
    "./Savestates/PGV_Pokecenter_Violet_city.state",
    "./Savestates/PGV_Route31_1.state",
    "./Savestates/PGV_Route31_2.state",
    "./Savestates/PGV_Route31_3.state",
    "./Savestates/PGV_Route31_4.state", 
    "./Savestates/PGV_Violet_city.state",
    "./Savestates/PGV_Knofensaturm_1.state",
    "./Savestates/PGV_Knofensaturm_2.state",
    "./Savestates/PGV_Knofensaturm_Boss.state",
    "./Savestates/PGV_Gym_boss.state",
    "./Savestates/PGV_Gym1.state",
    "./Savestates/PGV_Route32_1.state",
    "./Savestates/PGV_Route32_2.state", #before and after egg
    "./Savestates/PGV_Route32_3.state", #before and after egg
    "./Savestates/PGV_Route32_4.state",
    "./Savestates/PGV_Route32_5.state",

"""
Savestatepath=SAVE_STATE_PATHS[-6]
#Savestatepath="./Savestates/PGV.state"

# ──────────────────────────────────────────────────────────────
#  MODELL LADEN
# ──────────────────────────────────────────────────────────────

def get_path() -> str:
    candidates = [
    "./checkpoints/pokemon_ppo_42000000_steps.zip",
    #"./models/base_model.zip",
    #"./models/pokemon_ppo_final.zip",
    MODEL_SAVE_PATH + ".zip",
    MODEL_SAVE_PATH + "_continued.zip",
    ]

    # Neueste Checkpoints suchen
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
            return path
        else:
            return ""

def load_model() -> PPO:
    """
    Lädt das trainierte PPO-Modell.

    Suchreihenfolge:
      1. Bestes Modell (beste Evaluation während Training)
      2. Finales Modell (Ende des Trainings)
      3. Neuester Checkpoint
    """

    return PPO.load(get_path())



# ──────────────────────────────────────────────────────────────
#  MODUS 1: DIREKTE AUFNAHME
# ──────────────────────────────────────────────────────────────

def record_episode(model: PPO, episode_num: int = 1, output_path: str = None) -> dict:
    """
    Nimmt eine Episode auf und schreibt Frames direkt in die Videodatei (Streaming).

    Warum Streaming statt Liste?
    ────────────────────────────
    Alle Frames im RAM zu halten braucht enorm viel Speicher:
      45.000 Frames × 640×576×3 Bytes ≈ 50 GB RAM!
    Streaming schreibt jeden Frame sofort auf die Festplatte →
    maximaler RAM-Verbrauch: ~1 Frame (~1 MB) statt die gesamte Episode.

    Args:
        model       : Das trainierte PPO-Modell
        episode_num : Episodennummer (für Anzeige)
        output_path : Ausgabepfad der MP4-Datei (wird vorher übergeben)

    Returns:
        stats (dict): Episodenstatistiken für den Dateinamen
    """
    print(f"\n  → Nehme Episode {episode_num} auf ({FRAMES_PER_ACTION} Frames/Schritt @ {VIDEO_FPS} FPS)...")
    print(f"     Streaming direkt nach: {output_path}")

    env = PokemonGoldEnv(render_mode=None, record_actions=True)

    # Savestatepath überschreibt den zufälligen Pool → Video startet immer
    # vom definierten Startpunkt (oben konfiguriert: Savestatepath = "...")
    if os.path.exists(Savestatepath):
        with open(Savestatepath, "rb") as f:
            env._save_state_pool = [(Savestatepath, f.read())]

    obs, info = env.reset()

    # Video-Writer sofort öffnen – Frames werden direkt geschrieben.
    # faststart: verschiebt das moov-Atom an den Dateianfang →
    # Video-Player kann sofort abspielen ohne die ganze Datei zu laden.
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=VIDEO_FPS,
        codec="libx264",
        quality=8,
        output_params=["-movflags", "faststart"],
    )

    frame_count = 0
    step        = 0
    total_r     = 0.0

    try:
        while step < MAX_RECORD_STEPS:
            # ── KI wählt Aktion ──────────────────────────────────
            action, _ = model.predict(obs, deterministic=False)
            action = int(action)
            env._action_history.append(action)
            button_str = env.actions[action]

            # ── Taste drücken + JEDEN Frame direkt schreiben ─────
            env.pyboy.button_press(button_str)

            for frame_idx in range(FRAMES_PER_ACTION):
                env.pyboy.tick()

                screen_rgba = env.pyboy.screen.ndarray
                screen_rgb  = cv2.cvtColor(screen_rgba, cv2.COLOR_RGBA2RGB)
                frame_big   = cv2.resize(
                    screen_rgb,
                    (160 * VIDEO_SCALE, 144 * VIDEO_SCALE),
                    interpolation=cv2.INTER_NEAREST
                )

                if SHOW_OVERLAY:
                    frame_big = _add_info_overlay(frame_big, info, step, total_r)

                writer.append_data(frame_big)   # direkt auf Festplatte
                frame_count += 1

            env.pyboy.button_release(button_str)

            # ── Env-Interna aktualisieren ─────────────────────────
            obs              = env._get_observation()
            env._last_action = action
            reward           = env._calculate_reward()
            terminated       = env._check_terminated()
            truncated        = env._steps >= MAX_STEPS_PER_EPISODE
            env._steps      += 1
            env._episode_reward += reward
            info             = env._get_info()
            env._update_prev_state()

            total_r += reward
            step    += 1

            if step % 200 == 0:
                maps  = info.get("visited_maps", 0)
                tiles = info.get("visited_tiles", 0)
                dist  = info.get("max_dist_from_start", 0.0)
                xp    = info.get("episode_xp", 0)
                print(f"    Schritt {step}/{MAX_RECORD_STEPS}  |  Frames: {frame_count}  |  "
                      f"Reward: {total_r:.2f}  |  Karten: {maps}  |  Felder: {tiles}  |  "
                      f"Dist: {dist:.0f}  |  XP: {xp}")

            if terminated or truncated:
                reason = "Ziel erreicht!" if terminated else "Zeitlimit"
                print(f"    Episode {episode_num} beendet nach {step} Schritten ({reason})")
                print(f"    Gesamtbelohnung:     {total_r:.2f}")
                print(f"    Besuchte Felder:     {info.get('visited_tiles', 0)}")
                print(f"    Gefangene Pokemon:   {info.get('caught_count', 0)}")
                print(f"    Aufgenommene Frames: {frame_count}")
                print(f"    Videolänge:          {frame_count/VIDEO_FPS:.1f} Sekunden")
                break
    finally:
        writer.close()
        env.close()

    return {
        "steps":    step,
        "reward":   total_r,
        "felder":   info.get("visited_tiles", 0),
        "gefangen": info.get("caught_count",  0),
    }


def _add_info_overlay(frame: np.ndarray, info: dict, step: int, total_reward: float) -> np.ndarray:
    """
    Fügt einen Info-Text auf den Frame ein (Schritt, Reward, Pokemon-Anzahl).

    Das macht das Video informativer – du siehst direkt was passiert.

    Args:
        frame        : RGB-Bild als numpy-Array
        info         : Info-Dictionary aus env.step()
        step         : Aktueller Schritt
        total_reward : Kumulierter Reward der Episode

    Returns:
        frame mit eingefügtem Text
    """
    frame = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Schriftgröße an Frame-Höhe anpassen, nicht an VIDEO_SCALE —
    # OpenCV rendert unter font_scale ~0.4 unscharf.
    # Mindestgröße 0.45 damit Text auch bei kleinen Videos scharf bleibt.
    frame_h    = frame.shape[0]
    font_scale = max(0.45, frame_h / 576 * 0.65)
    thickness  = max(1, int(font_scale * 2))

    # Halbtransparenter Hintergrundstreifen – 0.45 Deckkraft damit Spielgeschehen durchscheint
    overlay_h  = int(font_scale * 80)
    overlay    = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    color_w = (255, 255, 255)
    shadow  = (40,  40,  40)

    y1 = int(font_scale * 30)
    y2 = int(font_scale * 62)

    caught  = info.get("caught_count", 0)
    tiles   = info.get("visited_tiles", 0)
    maps    = info.get("visited_maps", 0)
    hp_r    = info.get("hp_ratio", 1.0)
    level   = info.get("level", 0)
    dist    = info.get("max_dist_from_start", 0.0)

    line1 = f"Schr.: {step:4d} Rew: {total_reward:+7.2f} Maps: {maps}"
    line2 = f"Flr: {tiles:4d} Lv: {level} Gef: {caught} HP: {hp_r*100:.0f}% D: {dist:.0f}"

    # Bei kleinen Frames: kein LINE_AA (wird unscharf), sonst mit Antialiasing
    line_type = cv2.LINE_AA if font_scale >= 0.5 else cv2.LINE_8

    for text, y in ((line1, y1), (line2, y2)):
        cv2.putText(frame, text, (6, y + 1), font, font_scale, shadow,   thickness + 1, line_type)
        cv2.putText(frame, text, (5, y),     font, font_scale, color_w,  thickness,     line_type)

    # Reward-Wert farblich hervorheben (gelb wenn positiv)
    # if total_reward > 0:
    #     r_text = f"{total_reward:+7.2f}"
    #     x_pos  = int(5 + cv2.getTextSize("Schritt: 2048   Reward: ", font, font_scale, thickness)[0][0])
    #     cv2.putText(frame, r_text, (x_pos, 22), font, font_scale, color_y, thickness, cv2.LINE_AA)

    return frame


def save_video(frames: list, output_path: str):
    """
    Speichert eine Liste von Frames als MP4-Video.

    Verwendet imageio + ffmpeg für gute Kompatibilität.
    Das Video ist direkt mit VLC, Windows Media Player etc. abspielbar.

    Args:
        frames      : Liste von RGB-Bildern [(H, W, 3), ...]
        output_path : Ausgabepfad (z.B. "videos/episode_1.mp4")
    """
    if not frames:
        print("  ⚠ Keine Frames zum Speichern!")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"  → Exportiere {len(frames)} Frames als Video...")
    print(f"     Ausgabe: {output_path}")
    print(f"     Auflösung: {frames[0].shape[1]}×{frames[0].shape[0]} px")
    print(f"     FPS: {VIDEO_FPS}")

    # imageio schreibt direkt MP4 (benötigt ffmpeg als Backend)
    writer = imageio.get_writer(
        output_path,
        fps=VIDEO_FPS,
        codec="libx264",     # Guter, weit verbreiteter Codec
        quality=8,           # 0-10 (10 = beste Qualität, größere Datei)
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()
    print(f"  ✓ Video gespeichert: {output_path}")


# ──────────────────────────────────────────────────────────────
#  MODUS 2: ACTION REPLAY (alternativ)
# ──────────────────────────────────────────────────────────────

def save_action_history(actions: list, output_path: str):
    """
    Speichert die Aktionssequenz einer Episode als numpy-Datei.

    Diese Datei kann später mit replay_actions() abgespielt werden.

    Args:
        actions     : Liste von Aktions-Indizes [0, 4, 1, 2, ...]
        output_path : Pfad für die .npy-Datei
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(actions, dtype=np.int32))
    print(f"  ✓ {len(actions)} Aktionen gespeichert: {output_path}")


def replay_actions_as_video(actions_path: str, video_output: str):
    """
    Lädt eine gespeicherte Aktionssequenz und spielt sie in einer
    neuen PyBoy-Instanz ab – dabei wird ein Video aufgenommen.

    Warum ist das nützlich?
    ────────────────────────
    Du kannst einen interessanten Run speichern und ihn später
    beliebig oft als Video exportieren – ohne das Modell neu laufen zu lassen!

    Args:
        actions_path : Pfad zur .npy-Aktionsdatei
        video_output : Ausgabepfad für das Video
    """
    print(f"\n  → Replay aus: {actions_path}")
    actions = np.load(actions_path).tolist()
    print(f"     {len(actions)} Aktionen geladen")

    # Neue PyBoy-Instanz starten
    pyboy = PyBoy(ROM_PATH, window="null")
    pyboy.set_emulation_speed(0)

    # Save-State laden
    if os.path.exists(Savestatepath):
        with open(Savestatepath, "rb") as f:
            state_bytes = f.read()
        pyboy.load_state(io.BytesIO(state_bytes))

    frames = []
    action_names = ACTIONS

    for i, action_idx in enumerate(actions):
        button = action_names[action_idx]
        pyboy.button(button)

        for _ in range(FRAMES_PER_ACTION):
            pyboy.tick()

        # Frame aufnehmen
        screen_rgba  = pyboy.screen.ndarray
        screen_rgb   = cv2.cvtColor(screen_rgba, cv2.COLOR_RGBA2RGB)
        frame_big    = cv2.resize(
            screen_rgb,
            (160 * VIDEO_SCALE, 144 * VIDEO_SCALE),
            interpolation=cv2.INTER_NEAREST
        )
        frames.append(frame_big)

        if i % 100 == 0:
            print(f"    Replaying: {i}/{len(actions)} Schritte...", end="\r")

    pyboy.stop()
    print(f"\n    Replay abgeschlossen.")

    save_video(frames, video_output)


# ──────────────────────────────────────────────────────────────
#  SAVE-STATE ERSTELLEN (Hilfsfunktion)
# ──────────────────────────────────────────────────────────────

def create_save_state_interactively():
    """
    Startet PyBoy interaktiv zum manuellen Spielen.

    - Existiert bereits ein Save-State (Start.state) → wird automatisch geladen,
      du spielst von dort weiter.
    - Existiert noch kein Save-State → Spiel startet vom ROM-Anfang.

    Wenn du das Fenster schließt, wird der aktuelle Zustand
    automatisch als Start.state gespeichert – kein extra Tastendruck nötig.

    Verwendung:
        python record.py --create-state

    Steuerung im Fenster (PyBoy Standard):
        Pfeiltasten    → Bewegung
        Z              → A-Taste
        X              → B-Taste
        Enter          → Start
        Rücktaste      → Select
        Escape         → Fenster schließen + State speichern
    """
    print("\n" + "═" * 60)
    print("  SAVE-STATE ERSTELLEN / FORTSETZEN")
    print("═" * 60)
    print(f"  ROM:     {ROM_PATH}")
    print(f"  Ausgabe: {Savestatepath}")
    print()

    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)

    # Vorhandenen State laden falls vorhanden
    if os.path.exists(Savestatepath):
        with open(Savestatepath, "rb") as f:
            pyboy.load_state(f)
        print(f"  ✓ Bestehender Save-State geladen: {Savestatepath}")
        print(f"  → Spiele weiter und schließe das Fenster wenn du fertig bist.")
    else:
        print(f"  ⚠ Kein Save-State gefunden – starte vom ROM-Anfang.")
        print(f"  → Spiele bis zum gewünschten Punkt und schließe dann das Fenster.")
    print()

    # Spielschleife – läuft bis das Fenster geschlossen wird.
    # tick() gibt True zurück solange das Spiel läuft,
    # und False wenn der Nutzer das Fenster schließt oder ESC drückt.
    try:
        while pyboy.tick():
            pass
    except Exception:
        pass   # Manche PyBoy-Versionen werfen beim Schließen eine Exception

    # State automatisch speichern sobald das Fenster geschlossen wird
    with open(Savestatepath, "wb") as f:
        pyboy.save_state(f)
    print(f"\n  ✓ Save-State gespeichert: {Savestatepath}")

    pyboy.stop()


# ──────────────────────────────────────────────────────────────
#  HAUPT-AUFNAHMEFUNKTION
# ──────────────────────────────────────────────────────────────

def make_video_filename(episode_stats: dict, episode_num: int, model_path: str) -> str:
    """
    Erstellt einen informativen Dateinamen aus den Episode-Statistiken.

    Format:
      ep001_steps2048_reward+12.34_felder046_gefangen2_model-best_model.mp4

    Felder:
      ep       → Episodennummer
      steps    → Anzahl Schritte in dieser Episode
      reward   → Kumulierter Reward (mit Vorzeichen)
      felder   → Besuchte einzigartige Tiles
      gefangen → Gefangene Pokemon
      model    → Name des verwendeten Modells (ohne Pfad und .zip)

    Ungültige Zeichen (z.B. +, .) werden durch sichere Zeichen ersetzt
    damit der Dateiname auf Windows, Mac und Linux funktioniert.
    """
    from datetime import datetime

    steps    = episode_stats.get("steps",    0)
    reward   = episode_stats.get("reward",   0.0)
    felder   = episode_stats.get("felder",   0)
    gefangen = episode_stats.get("gefangen", 0)

    # Modellname: nur Dateiname ohne Pfad und .zip
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # Sonderzeichen ersetzen die Dateinamen auf Windows problematisch machen
    model_name = model_name.replace(" ", "_")

    # Reward: + durch p, - durch m, . durch K (für Komma) ersetzen
    reward_str = f"{reward:+.2f}".replace("+", "p").replace("-", "m").replace(".", "K")

    # Zeitstempel damit niemals überschrieben wird
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"ep{episode_num:03d}"
        f"_steps{steps:04d}"
        f"_reward{reward_str}"
        f"_felder{felder:04d}"
        f"_gefangen{gefangen}"
        f"_{model_name}"
        f"_{ts}"
        f".mp4"
    )
    return filename


def record():
    """
    Führt die komplette Aufnahme durch:
    1. Modell laden
    2. N_RECORD_EPISODES Episoden aufnehmen
    3. Jede Episode mit informativem Dateinamen speichern
       (bestehende Videos werden NIE überschrieben)
    """
    print("\n" + "═" * 60)
    print("  POKEMON GOLD RL – VIDEO-AUFNAHME")
    print("═" * 60)
    print(f"  Episoden:        {N_RECORD_EPISODES}")
    print(f"  Max Schritte:    {MAX_RECORD_STEPS}")
    print(f"  Auflösung:       {160*VIDEO_SCALE}×{144*VIDEO_SCALE} px @ {VIDEO_FPS} FPS")
    print(f"  Ausgabeordner:   {VIDEO_PATH}")
    print("═" * 60)

    os.makedirs(VIDEO_PATH, exist_ok=True)

    print("\n  Lade Modell...")
    model = load_model()

    # load_model() gibt das Modell zurück, aber wir brauchen auch den Pfad für den Dateinamen.
    # Deshalb wiederholen wir die Pfad-Suche kurz:
    model_path = _find_model_path()

    # Alle Videos desselben Modells in einen eigenen Unterordner schreiben
    # (Name = Modellname) → bei mehreren Modellen sauber sortiert.
    #model_name = os.path.splitext(os.path.basename(model_path))[0].replace(" ", "_")
    out_dir    = os.path.join(VIDEO_PATH, get_path())
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Modell-Ordner:   {out_dir}")

    for ep in range(1, N_RECORD_EPISODES + 1):
        # Temporärer Dateiname während der Aufnahme
        tmp_path = os.path.join(out_dir, f"_recording_ep{ep:03d}_tmp.mp4")

        # Aufnahme mit direktem Streaming → kein RAM-Problem
        stats = record_episode(model, episode_num=ep, output_path=tmp_path)

        # Am Ende umbenennen mit den echten Statistiken
        filename   = make_video_filename(stats, ep, model_path)
        final_path = os.path.join(out_dir, filename)
        os.rename(tmp_path, final_path)
        print(f"  ✓ Video gespeichert: {final_path}")

    print(f"\n✓ Alle Videos gespeichert in: {out_dir}")
    print("  Öffne die MP4-Dateien mit einem beliebigen Video-Player.\n")


def _find_model_path() -> str:
    """Gibt den Pfad des geladenen Modells zurück (für den Dateinamen)."""
    candidates = [
        #"./models/best_model.zip",
        MODEL_SAVE_PATH + ".zip",
        MODEL_SAVE_PATH + "_continued.zip",
    ]
    if os.path.isdir(CHECKPOINT_PATH):
        ckpts = sorted(
            [f for f in os.listdir(CHECKPOINT_PATH) if f.endswith(".zip")],
            key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_PATH, f)),
            reverse=True,
        )
        candidates += [os.path.join(CHECKPOINT_PATH, f) for f in ckpts]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "unknown_model"


# ──────────────────────────────────────────────────────────────
#  EINSTIEGSPUNKT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Prüfen ob Save-State erstellt werden soll
    if len(sys.argv) > 1 and sys.argv[1] == "--create-state":
        create_save_state_interactively()
    else:
        record()
