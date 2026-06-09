"""
╔══════════════════════════════════════════════════════════════╗
║   POKEMON GOLD RL  ·  RAM-Diagnose Skript (debug_ram.py)    ║
╚══════════════════════════════════════════════════════════════╝

ZWECK: Alle RAM-Werte live verfolgen während du spielst.
       Ideal um Map-IDs zu bestimmen und alle Adressen zu verifizieren.

AUSFÜHREN:
  python debug_ram.py

STEUERUNG:
  Pfeiltasten  → Bewegung
  Z            → A-Taste
  X            → B-Taste
  Enter        → Start
  ESC          → Beenden

EMPFOHLENER LAUF:
  1. Von New Bark Town starten
  2. Langsam bis Route 31 laufen und bei JEDEM Kartenwechsel stoppen
  3. Map Bank + Map Number notieren
  4. Mindestens einen Wildkampf eingehen → D116 Kampfstatus prüfen

BEKANNTE KORREKTE ADRESSEN:
  DA00 = Map Bank     ← NEU (aus Internetquelle)
  DA01 = Map Number   ← NEU (aus Internetquelle)
  D116 = Battle Type  ← NEU (0=kein Kampf, 1=Wild, 2=Trainer)
  D20D = X Position   ✓ verifiziert
  D20E = Y Position   ✓ verifiziert
  D0FF/D100 = Gegner aktuell HP  ✓ verifiziert
  D101/D102 = Gegner max HP      ← NEU (aus Internetquelle)
  DA49 = Level                   ✓ verifiziert
  DA4C/DA4D = HP                 ✓ verifiziert
  DA4E/DA4F = Max HP             ✓ verifiziert
  DA32-DA34 = XP                 ✓ verifiziert
  DA22 = Party Anzahl            ✓ verifiziert
"""

import os
import sys
from pyboy import PyBoy
from config import ROM_PATH, SAVE_STATE_PATHS

sys.stdout.reconfigure(encoding='utf-8')

# ── Welchen Save State laden? ────────────────────────────────
# 0 = PGV.state (New Bark Town) ← empfohlen für Map-ID Tour
STATE_INDEX = 0


def load_state(pyboy: PyBoy):
    path = SAVE_STATE_PATHS[STATE_INDEX]
    if os.path.exists(path):
        with open(path, "rb") as f:
            pyboy.load_state(f)
        print(f"✓ Save State geladen: {path}\n")
    else:
        print(f"⚠ Save State nicht gefunden: {path}\n")


def read_all(pyboy):
    """Liest alle relevanten RAM-Werte auf einmal."""

    # ── Karte (NEU – aus Internetquelle) ─────────────────────
    map_bank   = pyboy.memory[0xDA00]   # Map Bank (Gruppe)
    map_number = pyboy.memory[0xDA01]   # Map Number (ID innerhalb der Bank)

    # ── Spieler-Position (verifiziert) ───────────────────────
    x = pyboy.memory[0xD20D] // 2
    y = pyboy.memory[0xD20E] // 2

    # ── Kampfstatus (NEU – aus Internetquelle) ───────────────
    # D116: 0 = kein Kampf, 1 = Wild, 2 = Trainer, höher = Special
    battle_type = pyboy.memory[0xD116]

    # ── Eigene HP (verifiziert) ──────────────────────────────
    hp     = (pyboy.memory[0xDA4C] << 8) | pyboy.memory[0xDA4D]
    max_hp = (pyboy.memory[0xDA4E] << 8) | pyboy.memory[0xDA4F]

    # ── Level & XP (verifiziert) ─────────────────────────────
    level = pyboy.memory[0xDA49]
    xp    = (pyboy.memory[0xDA32] << 16) | (pyboy.memory[0xDA33] << 8) | pyboy.memory[0xDA34]

    # ── Team-Größe (verifiziert) ─────────────────────────────
    party = pyboy.memory[0xDA22]
    if party > 6:
        party = 0

    # ── Gegner-HP (verifiziert + maxHP NEU) ──────────────────
    enemy_hp     = (pyboy.memory[0xD0FF] << 8) | pyboy.memory[0xD100]
    enemy_max_hp = (pyboy.memory[0xD101] << 8) | pyboy.memory[0xD102]

    return (map_bank, map_number, x, y,
            battle_type,
            hp, max_hp, level, xp, party,
            enemy_hp, enemy_max_hp)


if __name__ == "__main__":
    print("═" * 85)
    print("  POKEMON GOLD – RAM LIVE TRACKER  (Neue verifizierte Adressen)")
    print("  Laufe von New Bark Town bis Route 31 – alle Werte werden geloggt")
    print("═" * 85)
    print()

    pyboy = PyBoy(ROM_PATH, window="SDL2")
    pyboy.set_emulation_speed(1)
    load_state(pyboy)

    # RAM nach State-Laden stabilisieren (erste ~60 Frames sind Übergangswerte)
    for _ in range(60):
        pyboy.tick()

    # Spaltenüberschriften
    print(f"  {'Bank':>4} {'Map':>4}   {'X':>4} {'Y':>4}   {'Kampf':>5}   {'HP':>9}  {'Lv':>3}  {'XP':>8}  {'Pty':>3}   {'GegHP':>9}")
    print(f"  {'DA00':>4} {'DA01':>4}   {'D20D':>4} {'D20E':>4}   {'D116':>5}   {'DA4C-F':>9}  {'DA49':>3}  {'DA32-4':>8}  {'DA22':>3}   {'D0FF-2':>9}")
    print("  " + "─" * 83)

    prev = None

    try:
        while pyboy.tick():
            current = read_all(pyboy)
            if current != prev:
                (map_bank, map_number, x, y,
                 battle_type,
                 hp, max_hp, level, xp, party,
                 enemy_hp, enemy_max_hp) = current

                hp_str  = f"{hp}/{max_hp}"
                geg_str = f"{enemy_hp}/{enemy_max_hp}" if enemy_hp > 0 else "—"

                # Kampfstatus-Label
                if battle_type == 0:
                    bat_str = "—"
                elif battle_type == 1:
                    bat_str = "Wild"
                elif battle_type == 2:
                    bat_str = "Train"
                else:
                    bat_str = f"#{battle_type}"

                print(f"  {map_bank:>4} {map_number:>4}   {x:>4} {y:>4}   {bat_str:>5}   "
                      f"{hp_str:>9}  {level:>3}  {xp:>8}  {party:>3}   {geg_str:>9}")
                prev = current

    except Exception as e:
        print(f"\n  Fehler: {e}")

    pyboy.stop()
    print("\n✓ Beendet.")
