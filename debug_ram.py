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
  D57C = Johto-Orden (Bitfeld)   ← NEU aus Quelle: $01=Falkner … $80=Clair (Gold/Silber!)
  D857 = Johto-Orden (Crystal!)  ← bisheriger Wert – das ist die CRYSTAL-Adresse, falsch für Gold
"""

import os
import sys
from pyboy import PyBoy
from config import ROM_PATH, SAVE_STATE_PATHS
from global_coords import GlobalCoordinateTransform
from pokemon_env import route_progress

sys.stdout.reconfigure(encoding='utf-8')

# Globaler Coord-Tracker – IDENTISCH zum Env (gleiche MAP_OFFSETS). Liefert gx/gy;
# daraus route_progress() = exakt der prog-Wert, den die Gain-/Frontier-Logik sieht.
_GCT = GlobalCoordinateTransform()


# ── Welchen Save State laden? ────────────────────────────────
# 0 = PGV.state (New Bark Town) ← empfohlen für Map-ID Tour
# -1 = letzter State (= Gym1/Arena) ← ideal für den ORDEN-Test:
#      Falkner besiegen und schauen, ob/welches Byte kippt.


final_path=SAVE_STATE_PATHS[-1]
# Für den Torhaus/Naht-Test: in Violet City starten (direkt am Torhaus → Route 31).
# Beliebig umstellen (z.B. PGV_Route31_1.state), je nachdem welche Naht du prüfst.
final_path="./Savestates/PGV_Route32_2.state"


def load_state(pyboy: PyBoy):
    path = final_path
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

    # ── Johto-Orden: ZWEI Kandidaten-Adressen parallel testen ─
    #   0xD57C = aus Internet-Quelle (Bit0 $01=Falkner … $80=Clair). Passt zur
    #            GOLD/SILBER-WRAM-Karte – wie deine übrigen Adressen (Party 0xDA22).
    #   0xD857 = bisheriger Wert – das ist aber die CRYSTAL-Adresse (≠ Gold).
    # Beim 1. Orden muss das RICHTIGE Byte von 0 → 1 (Bit0) springen.
    badge_d57c = pyboy.memory[0xD57C]
    badge_d857 = pyboy.memory[0xD857]

    # ── Ei im Team? Spezies-Liste ab 0xDA23, Ei = Marker 0xFD (253, "EGG") ─
    has_egg = int(0xFD in [pyboy.memory[0xDA23 + i] for i in range(party)])

    # ── Globale Weltkoordinaten + Pfad-Fortschritt (route_progress) ──────────
    # Exakt wie im Env: to_global(bank,num,x,y) → gx,gy → route_progress(leg).
    # An einer NAHT müssen die prog-Werte beider Seiten ≤2 auseinanderliegen,
    # sonst feuert die >2-Re-Baseline (= Door-Farm!). None = Innenraum/ungemappt.
    g = _GCT.to_global(map_bank, map_number, x, y)
    if g is None:
        glx = gly = None
        prog = None
    else:
        glx, gly = g
        prog = route_progress((map_bank, map_number), glx, gly)

    return (map_bank, map_number, x, y,
            battle_type,
            hp, max_hp, level, xp, party,
            enemy_hp, enemy_max_hp,
            badge_d57c, badge_d857, has_egg,
            glx, gly, prog)


# ── Badge-Region-Watcher (zeigt beim Orden-Erhalt, WELCHES Byte kippt) ──
# Beim Orden-Erhalt kippt GENAU EIN Byte von 0 auf 1. Wir beobachten das
# Fenster rund um den vielversprechenden Kandidaten 0xD57C (Internet-Quelle).
BADGE_WATCH_START = 0xD57A
BADGE_WATCH_END   = 0xD580   # exklusiv → 0xD57A..0xD57F (enthält 0xD57C)

def read_badge_window(pyboy):
    return tuple(pyboy.memory[a] for a in range(BADGE_WATCH_START, BADGE_WATCH_END))


# ── Party-Spezies-Watcher (Ei-Erkennung) ───────────────────────────────
# Ein Ei sitzt als Team-Slot mit Spezies-Marker 0xFD (253, "EGG") in der
# Party-Spezies-Liste ab 0xDA23. „Ei vorhanden" = 0xFD in der Liste. Der
# Watcher zeigt die Liste bei jeder Änderung → du siehst 0xFD auftauchen
# (oder den echten Ei-Wert, falls dein ROM einen anderen Marker nutzt).
def read_party_species_raw(pyboy):
    n = pyboy.memory[0xDA22]
    if n > 6:
        n = 0
    return tuple(pyboy.memory[0xDA23 + i] for i in range(n))


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
    print(f"  {'Bank':>4} {'Map':>4}   {'X':>4} {'Y':>4}   {'glX':>5} {'glY':>5} {'prog':>6}   {'Kampf':>5}   {'HP':>9}  {'Lv':>3}  {'XP':>8}  {'Pty':>3}   {'GegHP':>9}   {'Orden':>12} {'Orden':>12}  {'Ei':>4}")
    print(f"  {'DA00':>4} {'DA01':>4}   {'D20D':>4} {'D20E':>4}   {'glob':>5} {'glob':>5} {'leg':>6}   {'D116':>5}   {'DA4C-F':>9}  {'DA49':>3}  {'DA32-4':>8}  {'DA22':>3}   {'D0FF-2':>9}   {'D57C':>12} {'D857':>12}  {'0xFD':>4}")
    print("  " + "─" * 116)

    prev = None
    prev_badge_window = None
    prev_species = None

    try:
        while pyboy.tick():
            # ── Badge-Region separat überwachen → findet/bestätigt die Adresse ──
            bw = read_badge_window(pyboy)
            if prev_badge_window is not None and bw != prev_badge_window:
                print("  " + "·" * 70)
                print("  🏅 BADGE-REGION HAT SICH GEÄNDERT (besiegst du gerade Falkner?):")
                for off, val in enumerate(bw):
                    old = prev_badge_window[off]
                    if val != old:
                        addr = BADGE_WATCH_START + off
                        print(f"       0x{addr:04X}:  {old:3d} → {val:3d}    "
                              f"{old:08b} → {val:08b}    <== HIER kippt ein Bit")
                print("  " + "·" * 70)
            prev_badge_window = bw

            # ── Party-Spezies separat überwachen → Ei-Erkennung bestätigen ──
            sp = read_party_species_raw(pyboy)
            if prev_species is not None and sp != prev_species:
                print("  " + "·" * 70)
                print(f"  🥚 PARTY-SPEZIES GEÄNDERT (Ei abgeholt?):  Liste ab 0xDA23 = {list(sp)}")
                print(f"       Ei-Marker 0xFD (253) vorhanden: {'JA' if 0xFD in sp else 'nein'}")
                print("  " + "·" * 70)
            prev_species = sp

            current = read_all(pyboy)
            if current != prev:
                (map_bank, map_number, x, y,
                 battle_type,
                 hp, max_hp, level, xp, party,
                 enemy_hp, enemy_max_hp,
                 badge_d57c, badge_d857, has_egg,
                 glx, gly, prog) = current

                hp_str   = f"{hp}/{max_hp}"
                geg_str  = f"{enemy_hp}/{enemy_max_hp}" if enemy_hp > 0 else "—"
                b57c_str = f"{badge_d57c:08b}({bin(badge_d57c).count('1')})"
                b857_str = f"{badge_d857:08b}({bin(badge_d857).count('1')})"
                ei_str   = "JA" if has_egg else "—"
                glx_str  = f"{glx:>5}" if glx is not None else f"{'-':>5}"
                gly_str  = f"{gly:>5}" if gly is not None else f"{'-':>5}"
                prog_str = f"{prog:6.1f}" if prog is not None else f"{'-':>6}"

                # Kampfstatus-Label
                if battle_type == 0:
                    bat_str = "—"
                elif battle_type == 1:
                    bat_str = "Wild"
                elif battle_type == 2:
                    bat_str = "Train"
                else:
                    bat_str = f"#{battle_type}"

                print(f"  {map_bank:>4} {map_number:>4}   {x:>4} {y:>4}   {glx_str} {gly_str} {prog_str}   {bat_str:>5}   "
                      f"{hp_str:>9}  {level:>3}  {xp:>8}  {party:>3}   {geg_str:>9}   {b57c_str:>12} {b857_str:>12}  {ei_str:>4}")
                prev = current

    except Exception as e:
        print(f"\n  Fehler: {e}")

    pyboy.stop()
    print("\n✓ Beendet.")
