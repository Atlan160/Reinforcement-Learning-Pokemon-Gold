# 🏓 Pong RL – Lerne Reinforcement Learning am Beispiel

Ein vollständiges Pong-Spiel mit **PPO (Proximal Policy Optimization)**,
das zum Erlernen und Experimentieren mit Reinforcement Learning gedacht ist.

---

## 📁 Projektstruktur

```
pong_rl/
│
├── config.py       ← Alle Parameter (hier experimentieren!)
├── pong_game.py    ← Spielphysik (Ball, Schläger, Reflexion)
├── pong_env.py     ← Gymnasium-Umgebung (RL-Interface)
├── train.py        ← PPO-Training mit parallelen Envs
├── play.py         ← Spielen gegen den trainierten Agenten
└── requirements.txt
```

---

## 🚀 Schnellstart

### 1. Installation

```bash
# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Pakete installieren
pip install -r requirements.txt
```

### 2. Training starten

```bash
python train.py
```

Das Training dauert je nach Hardware **5–20 Minuten**.
Fortschritt wird live angezeigt. Modelle werden automatisch gespeichert.

### 3. Gegen den Agenten spielen

```bash
python play.py
```

Steuerung: **↑/↓** bewegen, **Q/E** kippen, **R** neu starten, **ESC** beenden.

### 4. Training visualisieren (optional)

```bash
tensorboard --logdir ./tensorboard_logs
# → Browser öffnen: http://localhost:6006
```

---

## 🎮 Spielmechanik

### Standard Pong
- Ball prallt von Wänden und Schlägern ab
- Wer den Ball am Gegner vorbeischickt, gewinnt den Punkt

### Kippen (TILTING_ENABLED = True)
Du (und der Agent) kannst deinen Schläger kippen!

```
Normaler Schläger:     Gekippter Schläger:
│                      ╱
│   →  ●               ╱   →  ●
│                      ╱           ↘  (abgelenkt)
```

Der Kippwinkel ändert den **Normalvektor** der Schlägeroberfläche
und damit die Abprallrichtung des Balls (siehe `pong_game.py`).

---

## 🧠 Reinforcement Learning – Konzepte

### Was lernt der Agent?

Der Agent lernt durch **Trial & Error** (Versuch und Irrtum):
1. Er probiert zufällige Aktionen aus
2. Manche führen zu Belohnungen (+1 für Punkt, -1 für Gegentreffer)
3. Er lernt, belohnende Aktionen in ähnlichen Situationen zu wiederholen

### Bestandteile

| Begriff | Bedeutung im Kontext |
|---------|---------------------|
| **Zustand (State)** | Ball-Position, -Geschwindigkeit, Schläger-Positionen |
| **Aktion (Action)** | Nichts / Hoch / Runter / Kippen CCW / Kippen CW |
| **Belohnung (Reward)** | +1.0 (Punkt), −1.0 (Gegentreffer), +0.05 (Treffer) |
| **Policy π** | Das neuronale Netz: P(Aktion \| Zustand) |
| **Episode** | Ein Spiel bis zum ersten Punkt |

### Warum PPO?

PPO ist der "Goldstandard" für diskrete Aktionsräume:
- **Stabil**: Clipping verhindert zu große Policy-Sprünge
- **Effizient**: Lernt mehrfach aus denselben Erfahrungen
- **Einfach**: Wenige sensible Hyperparameter
- **Bewährt**: Von OpenAI und vielen anderen eingesetzt

### Das Netzwerk

```
Eingabe (6 oder 8 Werte)
    ↓
[256 Neuronen] – Versteckte Schicht 1   ← NET_ARCH in config.py
    ↓
[256 Neuronen] – Versteckte Schicht 2
    ↙           ↘
Actor-Kopf    Critic-Kopf
P(a|s)        V(s)
(5 Ausgaben)  (1 Ausgabe)
```

**Actor**: Wählt Aktionen (Softmax → Wahrscheinlichkeiten)
**Critic**: Bewertet Zustände (hilft beim Berechnen des Advantage)

---

## ⚗️ Experimentieren

### Einfache Experimente (in `config.py`)

```python
# Kippfunktion an/aus
TILTING_ENABLED = False   # Klassisches Pong

# Bot schwieriger machen
BOT_DIFFICULTY = 0.95     # Sehr schwer → Agent braucht länger

# Mehr Trainingszeit
TOTAL_TIMESTEPS = 5_000_000

# Kleineres / größeres Netz
NET_ARCH = [128, 128]     # Kleiner, schneller
NET_ARCH = [512, 512, 256] # Größer, leistungsfähiger (aber langsamer)

# Mehr parallele Envs (falls mehrere CPU-Kerne vorhanden)
N_ENVS = 16
```

### Mittelschwere Experimente

**Belohnungsdesign ändern** (in `pong_game.py`, `_handle_collisions`):
```python
# Bonus für Ball in der Nähe der Gegner-Seite
if self.ball_x > WINDOW_WIDTH * 0.7:
    reward += 0.01
```

**Observations erweitern** (in `pong_game.py`, `get_observation`):
```python
# Ball-Richtung zum Agenten-Schläger hinzufügen
time_to_reach = (AGENT_PADDLE_X - self.ball_x) / max(abs(self.ball_vx), 0.1)
obs.append(np.clip(time_to_reach / 100, -1, 1))
```

**Bot-Verhalten ändern** (in `pong_game.py`, `apply_bot_action`):
```python
# Bot mit Vorhersage (schwieriger zu schlagen)
predicted_y = self.ball_y + self.ball_vy * (PLAYER_PADDLE_X - self.ball_x) / max(abs(self.ball_vx), 1)
diff = predicted_y - self.player_y
```

### Fortgeschrittene Experimente

- **Curriculum Learning**: Bot wird mit der Zeit besser (`BOT_DIFFICULTY` erhöhen während Training)
- **Self-Play**: Zwei PPO-Agenten gegeneinander trainieren
- **Continuous Actions**: `spaces.Box` statt `spaces.Discrete` für fließende Bewegung
- **CNN-Policy**: Pixel statt Zustandsvektor als Eingabe (`render_mode="rgb_array"`)

---

## 🐛 Häufige Probleme

**"Kein Modell gefunden"**
→ Zuerst `python train.py` ausführen

**Agent bewegt sich nicht / nur nach oben**
→ Training zu kurz. Mehr Schritte in `TOTAL_TIMESTEPS` oder neu trainieren.

**Agent lernt nicht (Belohnung bleibt bei ~−0.5)**
→ `BOT_DIFFICULTY` reduzieren (z.B. 0.6), damit Agent mehr Punkte macht.

**Training instabil (Belohnung springt stark)**
→ `LEARNING_RATE` halbieren, `CLIP_RANGE` reduzieren (z.B. 0.1)

**Pygame-Fehler beim Training**
→ Normal: Training läuft ohne Fenster. `render_mode=None` ist korrekt.

---

## 📚 Weiterführende Ressourcen

- **PPO Paper**: Schulman et al. 2017 – "Proximal Policy Optimization Algorithms"
- **Spinning Up**: https://spinningup.openai.com (OpenAI's RL-Einstiegskurs)
- **SB3 Docs**: https://stable-baselines3.readthedocs.io
- **Gymnasium Docs**: https://gymnasium.farama.org
