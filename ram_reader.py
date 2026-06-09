"""
╔══════════════════════════════════════════════════════════════╗
║   POKEMON GOLD RL  ·  RAM-Adressen auslesen (ram_reader.py) ║
╚══════════════════════════════════════════════════════════════╝

WAS IST DAS HIER?
──────────────────
Der GameBoy Color hat 8 KB internen RAM (0xC000 – 0xDFFF).
Pokemon Gold speichert ALLE Spielzustände dort:
  • Spielerposition, aktuelles Level
  • HP, Pokémon im Team, gefangene Pokémon
  • Geld, Items, Abzeichen
  • ...und vieles mehr.

Wir lesen diesen RAM aus und geben der KI strukturierte Infos
ZUSÄTZLICH zum Spielbild (CNN-Input). Das ist die hybride Strategie:

  ┌─ CNN (84×84 Bild) ──────────────────────────────────────┐
  │  "Was sehe ich gerade auf dem Bildschirm?"              │
  │  Erkennt: Kämpfe, Menüs, Terrain, Gegner, Dialoge       │
  └──────────────────────────────────────────────────────────┘
       +
  ┌─ RAM-Features (11 Zahlen) ───────────────────────────────┐
  │  "Wo bin ich? Wie geht es mir? Was habe ich erreicht?"   │
  │  Stabile, präzise Werte ohne Bildrauschen                │
  └──────────────────────────────────────────────────────────┘

WICHTIGER HINWEIS ZU DEN RAM-ADRESSEN:
────────────────────────────────────────
Die hier verwendeten Adressen stammen aus dem öffentlichen
Pokemon Gold/Silver Disassembly-Projekt (github.com/pret/pokegold).
Sie sind für die ENGLISCHE Pokemon Gold Version gedacht.

Falls dein ROM eine andere Region oder Version ist, können die
Adressen abweichen! Nutze dann die Hilfsfunktion dump_ram_region()
um selbst nach Adressen zu suchen.

Wie erkenne ich die richtige Adresse?
1. Spiele das Spiel manuell ein paar Schritte
2. Nutze dump_ram_region(pyboy, 0xD800, 0xD900)
3. Bewege dich und schaue welche Werte sich ändern
4. Die sich ändernden Adressen sind deine Kandidaten!
"""

import numpy as np
from pyboy import PyBoy

from global_coords import GlobalCoordinateTransform

import sys
sys.stdout.reconfigure(encoding='utf-8')


# ══════════════════════════════════════════════════════════════
#  RAM-ADRESSEN (Pokemon Gold, englische Version)
#  Quelle: https://github.com/pret/pokegold/blob/master/constants/wram_constants.asm
# ══════════════════════════════════════════════════════════════

class PokemonGoldRAM:
    """
    Zentrale Sammlung aller RAM-Adressen für Pokemon Gold.

    Alle Adressen wurden durch RAM-Scanning mit debug_ram.py ermittelt
    und mit bekannten Spielwerten verifiziert.

    LEGENDE:
      ✓ = eindeutig verifiziert (Wert beobachtet und bestätigt)
      ? = plausibler Kandidat (noch nicht 100% bestätigt)

    Verwendung:
        value = pyboy.memory[PokemonGoldRAM.PLAYER_X]
    """

    # ══════════════════════════════════════════════════════
    #  SPIELERPOSITION  (?)
    #  Verifiziert durch Kartenvergleich:
    #    Starterstadt:  0xD009=5,  0xD00A=5
    #    Route 29:      0xD009=1,  0xD00A=2
    #  Welches X und welches Y ist, muss noch durch
    #  gezieltes Bewegen (nur rechts oder nur hoch) geprüft werden.
    # ══════════════════════════════════════════════════════
    # ✓ Verifiziert durch Bewegungstest (PGV.state, New Bark Town):
    #   Rechts: X +2 pro Schritt, Y unveraendert
    #   Hoch:   Y -2 pro Schritt, X unveraendert
    #   Format: 2 Einheiten pro Tile (1 Tile = 16 Pixel = 2 Koordinaten-Einheiten)
    #   Startposition New Bark Town: X=9, Y=10
    PLAYER_X        = 0xD20D
    PLAYER_Y        = 0xD20E

    # Karten-Identifikation – verifiziert mit debug_ram.py + pokegold Disassembly
    # map_key = (MAP_BANK, MAP_NUMBER) → eindeutige Karten-ID (1-indiziert pro Gruppe)
    #
    # ✓ Verifizierte Karten-IDs (Bank, Map) → Disassembly-Name:
    #   (24, 3) ROUTE_29                ← cleared
    #   (24, 4) NEW_BARK_TOWN           ← cleared
    #   (24, 5) ELMS_LAB                ← trap
    #   (24, 6) PLAYERS_HOUSE_1F        ← trap
    #   (24, 7) PLAYERS_HOUSE_2F        ← trap (Schlupfloch wie Pokecenter-OG)
    #   (24, 8) PLAYERS_NEIGHBORS_HOUSE ← trap
    #   (24, 9) ELMS_HOUSE              ← trap
    #   (26, 1) ROUTE_30                ← langer Korridor (cleared ab Lv 13)
    #   (26, 2) ROUTE_31                ← Weg nach Violet City – VOLLER Reward!
    #   (26, 3) CHERRYGROVE_CITY        ← voller Bonus
    #   (26, 4) CHERRYGROVE_MART        ← trap
    #   (26, 5) CHERRYGROVE_POKECENTER_1F ← trap
    #   (20, 1) Pokecenter OG (gruppenübergreifend) ← trap
    #   (26, 6..8) Cherrygrove Speech-Häuser        ← trap
    #   (26, 9) ROUTE_30_BERRY_HOUSE    ← trap
    #   (26,10) MR_POKEMONS_HOUSE       ← trap
    #   (26,11) ROUTE_31_VIOLET_GATE    → Violet City (Gruppe 10)
    MAP_BANK        = 0xDA00   # ✓ Map Bank (Gruppe)
    MAP_NUMBER      = 0xDA01   # ✓ Map Number (ID innerhalb der Bank)

    # ══════════════════════════════════════════════════════
    #  TEAM: ANZAHL UND ARTEN  (✓ vollständig verifiziert)
    #
    #  Struktur im RAM ab 0xDA22:
    #    [party_count, species_0, species_1, ..., 0xFF, <mon1_struct>, <mon2_struct>, ...]
    #
    #  Verifiziert:
    #    1 Pokemon (Totodile):  count=1, species=[158, 0xFF]
    #    2 Pokemon (+Pidgey):   count=2, species=[158, 16, 0xFF]
    # ══════════════════════════════════════════════════════
    PARTY_COUNT          = 0xDA22   # ✓ Anzahl Pokemon im Team (0-6)
    PARTY_SPECIES_START  = 0xDA23   # ✓ Beginn der Species-Liste (7 Bytes)

    # ══════════════════════════════════════════════════════
    #  POKEMON 1 IM TEAM  (✓ vollständig verifiziert)
    #
    #  Struct-Größe: 48 Bytes (0x30)
    #  Struct-Start: 0xDA2A = PARTY_COUNT + 1 (count) + 7 (species list)
    #
    #  Verifiziert mit Totodile Level 8, HP 27→19:
    # ══════════════════════════════════════════════════════
    PARTY_MON1_SPECIES    = 0xDA2A   # ✓ Species-ID  (158 = Totodile)
    PARTY_MON1_EXP_HIGH   = 0xDA32   # ✓ XP Byte 2 (höchstwertiges, 3-Byte Big-Endian)
    PARTY_MON1_EXP_MID    = 0xDA33   # ✓ XP Byte 1
    PARTY_MON1_EXP_LOW    = 0xDA34   # ✓ XP Byte 0 (Level-3-Sentret gab +24 XP)
    PARTY_MON1_LEVEL      = 0xDA49   # ✓ Level       (war 8)
    PARTY_MON1_STATUS     = 0xDA4A   # ? Status-Byte (0=OK, >0=vergiftet/etc.)
    PARTY_MON1_HP_HIGH    = 0xDA4C   # ✓ Akt. HP High-Byte (Big-Endian)
    PARTY_MON1_HP_LOW     = 0xDA4D   # ✓ Akt. HP Low-Byte  (= 19 nach Schaden)
    PARTY_MON1_MAXHP_HIGH = 0xDA4E   # ✓ Max HP High-Byte
    PARTY_MON1_MAXHP_LOW  = 0xDA4F   # ✓ Max HP Low-Byte   (= 27)

    # ══════════════════════════════════════════════════════
    #  POKEMON 2 IM TEAM  (? berechnet, Species verifiziert)
    #
    #  Struct-Start = PARTY_MON1_STRUCT + 0x30 (48 Bytes)
    #  Species bei 0xDA5A = 16 (Pidgey) ✓ verifiziert
    # ══════════════════════════════════════════════════════
    PARTY_MON2_SPECIES    = 0xDA5A   # ✓ Species-ID  (16 = Pidgey)
    PARTY_MON2_LEVEL      = 0xDA79   # ? Level       (berechnet: 0xDA5A + 0x1F)
    PARTY_MON2_HP_HIGH    = 0xDA7C   # ? Akt. HP High-Byte (berechnet)
    PARTY_MON2_HP_LOW     = 0xDA7D   # ? Akt. HP Low-Byte
    PARTY_MON2_MAXHP_HIGH = 0xDA7E   # ? Max HP High-Byte (berechnet)
    PARTY_MON2_MAXHP_LOW  = 0xDA7F   # ? Max HP Low-Byte

    # ══════════════════════════════════════════════════════
    #  GEGNER-HP IM KAMPF  (✓ verifiziert)
    #
    #  Verifiziert durch Vor/Nachher-Vergleich mit debug_ram.py:
    #    Level-2-Sentret:  0xD100 = 13 (vor Angriff) → 1  (nach Angriff)
    #    Level-3-Sentret:  0xD100 = 15 (vor Kampf)   → 0  (nach Ohnmacht)
    #  High-Byte bleibt 0 solange Gegner < 256 HP hat (alle Wildpokemon).
    # ══════════════════════════════════════════════════════
    ENEMY_HP_HIGH      = 0xD0FF   # ✓ Gegner akt. HP High-Byte
    ENEMY_HP_LOW       = 0xD100   # ✓ Gegner akt. HP Low-Byte
    ENEMY_MAXHP_HIGH   = 0xD101   # ✓ Gegner max  HP High-Byte (aus Internetquelle)
    ENEMY_MAXHP_LOW    = 0xD102   # ✓ Gegner max  HP Low-Byte

    # ══════════════════════════════════════════════════════
    #  KAMPFTYP / KAMPFSTATUS  (✓ verifiziert mit debug_ram.py)
    #
    #  0xD116 = 0  → kein Kampf  (zeigt "—")
    #  0xD116 = 1  → Wildkampf   (zeigt "Wild") ← direkt beobachtet
    #  0xD116 = 2  → Trainerkampf (noch nicht getestet)
    #  höhere Werte = spezielle Kämpfe (Gym etc.)
    #
    #  Wichtig: Wechselt sofort auf 1 beim Kampfstart,
    #  ABER ENEMY_HP ist noch nicht geladen (Lazy-Init nötig!).
    #  Nach Flucht: D116 → 0, aber GegHP-RAM-Wert bleibt als
    #  Restwert stehen → niemals GegHP als Kampfindikator nutzen!
    # ══════════════════════════════════════════════════════
    BATTLE_TYPE     = 0xD116   # ✓ verifiziert

    # ══════════════════════════════════════════════════════
    #  NOCH NICHT GEFUNDEN  (Platzhalter)
    #  TODO: Mit debug_ram.py weitersuchen wenn diese
    #        Infos für das Reward-System benötigt werden.
    # ══════════════════════════════════════════════════════
    POKEDEX_CAUGHT_START = 0x0000   # TODO: Pokemon fangen, vor/nach vergleichen
    JOHTO_BADGES         = 0x0000   # TODO: Abzeichen gewinnen, vergleichen
    MONEY_HIGH           = 0x0000   # TODO: Einkauf tätigen, vergleichen
    MONEY_MID            = 0x0000   # TODO
    MONEY_LOW            = 0x0000   # TODO


# ══════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN zum Auslesen
# ══════════════════════════════════════════════════════════════

def read_hp(pyboy: PyBoy) -> tuple[int, int]:
    """
    Liest die aktuellen und maximalen HP des ersten Team-Pokemon.

    Gespeichert als Big-Endian 2-Byte-Wert:
      HP = (High-Byte << 8) | Low-Byte
      Beispiel: [0x00, 0x1B] → 27 HP  ✓ verifiziert
      Nach Schaden: [0x00, 0x13] → 19 HP  ✓ verifiziert

    Rückgabe: (aktuelle_HP, max_HP)
    """
    current_hp = (pyboy.memory[PokemonGoldRAM.PARTY_MON1_HP_HIGH] << 8) \
               |  pyboy.memory[PokemonGoldRAM.PARTY_MON1_HP_LOW]

    max_hp     = (pyboy.memory[PokemonGoldRAM.PARTY_MON1_MAXHP_HIGH] << 8) \
               |  pyboy.memory[PokemonGoldRAM.PARTY_MON1_MAXHP_LOW]

    return current_hp, max_hp


def read_exp(pyboy: PyBoy) -> int:
    """
    Liest die Erfahrungspunkte (XP) des ersten Team-Pokemon.

    Gespeichert als 3-Byte Big-Endian Wert (max. 1.000.000 in Gen 2):
      XP = (Byte2 << 16) | (Byte1 << 8) | Byte0

    Adressen (Offset +0x08 im Party-Mon-Struct ab 0xDA2A):
      0xDA32 = Byte 2 (höchstwertiges)
      0xDA33 = Byte 1
      0xDA34 = Byte 0 (niedrigstwertiges)

    Noch nicht live verifiziert – Adressen aus pokegold-Struct-Layout abgeleitet.
    """
    return (
        (pyboy.memory[PokemonGoldRAM.PARTY_MON1_EXP_HIGH] << 16)
      | (pyboy.memory[PokemonGoldRAM.PARTY_MON1_EXP_MID]  <<  8)
      |  pyboy.memory[PokemonGoldRAM.PARTY_MON1_EXP_LOW]
    )


def read_total_party_xp(pyboy: PyBoy) -> int:
    """
    Liest die Summe der XP ALLER Pokemon im Team.

    Warum nötig?
    Wenn ein gefangenes Pokemon (z.B. Pidgey in Slot 2) kämpft und XP
    bekommt, ändert sich nur PARTY_MON2_EXP – read_exp() (nur Slot 1)
    würde das komplett übersehen → kein XP-Reward für Kämpfe mit dem
    zweiten Pokemon.

    Struct-Layout (verifiziert):
      Struct-Größe:   0x30 Bytes pro Pokemon
      Struct-Start:   0xDA2A (Mon 1), 0xDA5A (Mon 2), 0xDA8A (Mon 3), ...
      XP-Offset:      +0x08 (3 Bytes, Big-Endian)

    Rückgabe: Summe aller XP im Team (int)
    """
    count = read_party_count(pyboy)
    total = 0
    for i in range(min(count, 6)):
        struct_start = 0xDA2A + i * 0x30
        xp = (
            (pyboy.memory[struct_start + 0x08] << 16) |
            (pyboy.memory[struct_start + 0x09] <<  8) |
             pyboy.memory[struct_start + 0x0A]
        )
        total += xp
    return total


def read_enemy_hp(pyboy: PyBoy) -> int:
    """
    Liest die aktuellen HP des gegnerischen Pokemon im Kampf.

    Gespeichert als Big-Endian 2-Byte-Wert:
      HP = (High-Byte << 8) | Low-Byte

    ✓ Verifiziert durch debug_ram.py Vor/Nachher-Vergleich:
      Level-2-Sentret: 13 HP vor Angriff → 1 HP nach Angriff
      Level-3-Sentret: 15 HP vor Kampf   → 0 HP nach Ohnmacht

    Außerhalb eines Kampfes ist der Wert bedeutungslos (0 oder Restwert).
    Daher nur im Kampf auslesen: read_in_battle() vorher prüfen!

    Rückgabe: aktuelle Gegner-HP (int)
    """
    return (
        (pyboy.memory[PokemonGoldRAM.ENEMY_HP_HIGH] << 8)
      |  pyboy.memory[PokemonGoldRAM.ENEMY_HP_LOW]
    )


def read_level(pyboy: PyBoy) -> int:
    """
    Liest das Level des ersten Team-Pokemon.

    ✓ Verifiziert: PARTY_MON1_LEVEL = 0xDA49 (war Level 8)

    Rückgabe: aktuelles Level (int, 1–100)
    """
    return pyboy.memory[PokemonGoldRAM.PARTY_MON1_LEVEL]


def read_player_position(pyboy: PyBoy) -> tuple[int, int, int, int]:
    """
    Liest die aktuelle Spielerposition.

    Rückgabe: (map_bank, map_number, x, y)
      map_bank   : Karten-Bank (Gruppe) – DA00
      map_number : Karten-Nummer – DA01
      x, y       : Tile-Koordinaten auf der aktuellen Karte

    map_key = (map_bank, map_number) → eindeutige Karten-ID für _visited_maps etc.

    Muss noch mit debug_ram.py verifiziert werden!
    """
    map_bank   = pyboy.memory[PokemonGoldRAM.MAP_BANK]
    map_number = pyboy.memory[PokemonGoldRAM.MAP_NUMBER]
    # X und Y sind in 2-Einheiten-pro-Tile gespeichert → durch 2 = Tile-Koordinate
    x          = pyboy.memory[PokemonGoldRAM.PLAYER_X] // 2
    y          = pyboy.memory[PokemonGoldRAM.PLAYER_Y] // 2
    return map_bank, map_number, x, y


def read_party_count(pyboy: PyBoy) -> int:
    """
    Liest die Anzahl der Pokemon im Team (0-6).

    ✓ Verifiziert:
      1 Pokemon (Totodile):        0xDA22 = 1
      2 Pokemon (+ Pidgey gefangen): 0xDA22 = 2
    """
    count = pyboy.memory[PokemonGoldRAM.PARTY_COUNT]
    # Sanity-Check: max. 6 Pokemon möglich
    return count if count <= 6 else 0


def read_party_species(pyboy: PyBoy) -> set:
    """
    Liest die Species-IDs aller Pokemon im Team als Set zurück.

    ✓ Verifiziert: PARTY_SPECIES_START = 0xDA23
      1 Pokemon (Totodile): {158}
      2 Pokemon (+Pidgey):  {158, 16}

    Nützlich für Diversitäts-Reward: neue Art im Team = echter Fangerfolg.
    0xFF ist das Listenende-Marker und wird ignoriert.
    """
    count = read_party_count(pyboy)
    species = set()
    for i in range(count):
        s = pyboy.memory[PokemonGoldRAM.PARTY_SPECIES_START + i]
        if s != 0xFF:
            species.add(s)
    return species


def read_caught_pokemon_count(pyboy: PyBoy) -> int:
    """
    Zählt wie viele verschiedene Pokemon bereits gefangen wurden.

    Der Pokedex ist als Bit-Array gespeichert:
    - 32 Bytes = 256 Bits
    - Bit i = Pokemon Nummer i+1 gefangen?

    Wir zählen alle gesetzten Bits = gefangene Pokemon.
    """
    # Pokedex-Adresse noch nicht gefunden (Platzhalter 0x0000).
    # TODO: Adresse per debug_ram.py nach dem Fangen eines Pokemon ermitteln.
    # Fallback: Party Count als Näherungswert (gefangene ≥ Team-Größe)
    if PokemonGoldRAM.POKEDEX_CAUGHT_START == 0x0000:
        return read_party_count(pyboy)

    total = 0
    for i in range(32):
        byte = pyboy.memory[PokemonGoldRAM.POKEDEX_CAUGHT_START + i]
        total += bin(byte).count('1')
    return total


def read_badge_count(pyboy: PyBoy) -> int:
    """
    Zählt die Gesamtzahl der Abzeichen (Johto + Kanto).

    Johto-Abzeichen: Bits 0-7 in Byte 0xD857
    Kanto-Abzeichen: Bits 0-7 in Byte 0xD858

    Bin(n).count('1') zählt die gesetzten Bits (= erhaltene Abzeichen).
    """
    # Abzeichen-Adresse noch nicht gefunden → 0 zurückgeben bis gefunden.
    if PokemonGoldRAM.JOHTO_BADGES == 0x0000:
        return 0
    johto = bin(pyboy.memory[PokemonGoldRAM.JOHTO_BADGES]).count('1')
    return johto


def read_money(pyboy: PyBoy) -> int:
    """
    Liest den aktuellen Geldbetrag in Yen (¥).

    Geld ist als BCD (Binary-Coded Decimal) gespeichert.
    BCD bedeutet: jede Dezimalziffer belegt 4 Bits.

    Beispiel: 1234 ¥
      → gespeichert als: 0x00, 0x12, 0x34
      → 0x00 = 00, 0x12 = 12, 0x34 = 34
      → Ergebnis: 001234 ¥

    Konvertierung BCD → Dezimal:
      (high_nibble × 10) + low_nibble
      wobei nibble = 4 Bits
    """
    # Geld-Adresse noch nicht gefunden → 0 zurückgeben bis gefunden.
    if PokemonGoldRAM.MONEY_HIGH == 0x0000:
        return 0

    def bcd_to_int(byte: int) -> int:
        return ((byte >> 4) * 10) + (byte & 0x0F)

    high = bcd_to_int(pyboy.memory[PokemonGoldRAM.MONEY_HIGH])
    mid  = bcd_to_int(pyboy.memory[PokemonGoldRAM.MONEY_MID])
    low  = bcd_to_int(pyboy.memory[PokemonGoldRAM.MONEY_LOW])

    # high × 10000 + mid × 100 + low (weil BCD in Schritten von 2 Ziffern)
    return high * 10000 + mid * 100 + low


def read_in_battle(pyboy: PyBoy) -> bool:
    """
    Gibt True zurück wenn aktuell ein Kampf läuft.

    Adresse 0xD116 (BATTLE_TYPE) – aus Internetquelle:
      0 = kein Kampf
      1 = Wildkampf
      2 = Trainerkampf
      höher = Spezialkampf (Gym etc.)

    Muss noch mit debug_ram.py verifiziert werden!

    Nützlich um:
    - Nur auf der Karte Belohnungen für Exploration zu geben
    - Im Kampf andere Rewards zu verwenden (z.B. HP-Verlust)
    """
    return pyboy.memory[PokemonGoldRAM.BATTLE_TYPE] != 0


def get_all_ram_features(pyboy: PyBoy, coord_transform=None) -> np.ndarray:
    """
    Liest ALLE relevanten RAM-Werte und gibt sie als normalisiertes numpy-Array zurück.

    NORMALISIERUNG ist wichtig für neuronale Netze!
    Warum? Neuronen arbeiten am besten mit Werten im Bereich [-1, 1] oder [0, 1].
    Ein unnormalisierter Wert wie "Geld = 50000" würde das Netz verwirren,
    weil andere Features nur 0-6 groß sind.

    Alle Features werden auf [0, 1] normalisiert:
      normierter_Wert = aktueller_Wert / maximaler_möglicher_Wert

    Rückgabe: numpy-Array mit N_RAM_FEATURES Einträgen (float32)

    ACHTUNG: Die Reihenfolge und Anzahl darf sich NICHT ändern nachdem
    du mit dem Training angefangen hast! Das Netz lernt "Feature 3 = HP-Ratio"
    und würde falsche Ergebnisse liefern wenn du Features umordnest.
    """
    map_bank, map_number, x, y  = read_player_position(pyboy)
    current_hp, max_hp          = read_hp(pyboy)
    party_count                 = read_party_count(pyboy)
    caught_count                = read_caught_pokemon_count(pyboy)
    badge_count                 = read_badge_count(pyboy)
    money                       = read_money(pyboy)
    in_battle                   = read_in_battle(pyboy)

    # HP-Verhältnis (0.0 = tot, 1.0 = volle HP)
    hp_ratio = current_hp / max(max_hp, 1)   # max(..., 1) verhindert Division durch 0

    # Globale Weltkoordinaten – ersetzen die rohen map_bank/map_number-Werte.
    # Ohne übergebene Instanz: frische (ohne Carry-Forward), nur für Debug-Aufrufe.
    if coord_transform is None:
        coord_transform = GlobalCoordinateTransform()
    global_x, global_y, is_indoor = coord_transform.features(map_bank, map_number, x, y)

    features = np.array([
        # --- Spielerposition (lokal + global) ---
        x / 255.0,                   # lokale X-Position [0, 1]
        y / 255.0,                   # lokale Y-Position [0, 1]
        global_x,                    # globale Welt-X-Position [0, 1] (gestitcht)
        global_y,                    # globale Welt-Y-Position [0, 1] (gestitcht)
        is_indoor,                   # 1.0 = Innenraum/unbekannte Karte, 0.0 = Oberwelt

        # --- Zustand des Teams ---
        hp_ratio,                    # HP-Verhältnis [0.0, 1.0]
        party_count / 6.0,           # Team-Größe normiert [0.0, 1.0]

        # --- Fortschritt ---
        caught_count / 251.0,        # Gefangene Pokemon [0.0, 1.0]
        badge_count / 16.0,          # Abzeichen [0.0, 1.0] (16 total)
        min(money, 999_999) / 999_999.0,  # Geld normiert [0.0, 1.0]

        # --- Kampfstatus ---
        float(in_battle),            # 0.0 = keine Kampf, 1.0 = Kampf aktiv

    ], dtype=np.float32)

    return features


# ══════════════════════════════════════════════════════════════
#  DEBUG-HILFSFUNKTIONEN
#  Nutze diese um neue RAM-Adressen zu entdecken!
# ══════════════════════════════════════════════════════════════

def dump_ram_region(pyboy: PyBoy, start: int, end: int) -> dict:
    """
    Gibt alle RAM-Werte in einem Bereich als Dictionary zurück.

    Verwendung zum Finden neuer Adressen:

    1. Spiele manuell bis zu einem interessanten Zustand
    2. Rufe dump_ram_region(pyboy, 0xDC00, 0xDD00) auf → speichere als dict_a
    3. Ändere im Spiel etwas (z.B. Geld ausgeben)
    4. Rufe dump_ram_region erneut auf → speichere als dict_b
    5. Vergleiche: {k: (dict_a[k], dict_b[k]) for k in dict_a if dict_a[k] != dict_b[k]}
    6. Die geänderten Adressen sind deine Kandidaten!

    Beispiel:
        state_before = dump_ram_region(pyboy, 0xD840, 0xD860)
        # ... kaufe etwas im Spiel ...
        state_after  = dump_ram_region(pyboy, 0xD840, 0xD860)
        changes = {k: (state_before[k], state_after[k])
                   for k in state_before
                   if state_before[k] != state_after[k]}
        print("Geänderte Adressen:", changes)
    """
    return {
        hex(addr): pyboy.memory[addr]
        for addr in range(start, end)
    }


def print_game_state(pyboy: PyBoy):
    """
    Gibt den aktuellen Spielzustand lesbar in der Konsole aus.
    Nützlich zum Debuggen während des Trainings.

    Beispiel-Ausgabe:
        === Pokemon Gold – Spielzustand ===
        Position:  Map 01-04 | X=14 | Y=22
        Team:      2 Pokemon im Team
        HP:        38/50 (76.0%)
        Gefangen:  5 Pokemon im Pokédex
        Abzeichen: 1
        Geld:      1200 ¥
        Kampf:     Nein
    """
    map_bank, map_number, x, y  = read_player_position(pyboy)
    current_hp, max_hp          = read_hp(pyboy)
    exp                         = read_exp(pyboy)
    party_count                 = read_party_count(pyboy)
    caught_count                = read_caught_pokemon_count(pyboy)
    badge_count                 = read_badge_count(pyboy)
    money                       = read_money(pyboy)
    in_battle                   = read_in_battle(pyboy)
    enemy_hp                    = read_enemy_hp(pyboy) if in_battle else 0

    hp_pct = (current_hp / max(max_hp, 1)) * 100

    print("=== Pokemon Gold – Spielzustand ===")
    print(f"  Position:  Map ({map_bank},{map_number}) | X={x} | Y={y}")
    print(f"  Team:      {party_count} Pokemon im Team")
    print(f"  HP:        {current_hp}/{max_hp} ({hp_pct:.1f}%)")
    print(f"  XP:        {exp}")
    print(f"  Gefangen:  {caught_count} Pokemon im Pokédex")
    print(f"  Abzeichen: {badge_count}")
    print(f"  Geld:      {money} ¥")
    print(f"  Kampf:     {'Ja' if in_battle else 'Nein'}")
    if in_battle:
        print(f"  Gegner-HP: {enemy_hp}")
    print()
