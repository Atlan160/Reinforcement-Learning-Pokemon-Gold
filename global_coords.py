"""
╔══════════════════════════════════════════════════════════════╗
║  POKEMON GOLD RL  ·  Globale Weltkoordinaten (global_coords) ║
╚══════════════════════════════════════════════════════════════╝

PROBLEM
────────
Die X/Y-Koordinaten aus dem RAM gelten nur LOKAL pro Karte und springen
bei jedem Kartenwechsel. Die KI kann daraus kein Gefühl für "Nähe" oder
"Richtung" entwickeln ("geh nach Norden = Fortschritt").

LÖSUNG: SEAM-STITCHING
───────────────────────
Wir legen pro Karte einen festen Offset fest und kleben die Karten zu
einer durchgehenden Weltebene zusammen:

    global_x = offset_x[map] + local_x
    global_y = offset_y[map] + local_y

Die Offsets stammen aus einem debug_ram.py Walk-Through: An jeder
Kartengrenze schaut man, welche lokale Position direkt vor und direkt
nach dem Übergang gelesen wird, und kettet die Karten daran an.
Anker = New Bark Town bei (0, 0).

VALIDIERUNG (Walk-Through New Bark → Norden Route 31):
  Beim geraden Marsch nach Norden (Cherrygrove → Route 30 → Route 31)
  bleibt global_x konstant bei ≈ -42 und global_y sinkt monoton.
  → Die Naht-Offsets sind konsistent.

KOORDINATENEINHEIT
───────────────────
Gleiche Einheit wie read_player_position() (= RAM-Wert // 2). Solange
diese Funktion und die Offsets dieselbe Einheit nutzen, ist das Stitching
korrekt – die exakte Tile/Block-Zuordnung spielt keine Rolle.

INNENRÄUME & FRONTIER
──────────────────────
Gebäude/Höhlen/noch-nicht-gestitchte Karten haben keinen Offset.
Dort gilt Carry-Forward: die letzte bekannte Außenposition wird
beibehalten (kein Sprung nach (0,0)) und das is_indoor-Flag = 1 gesetzt.
Das CNN unterscheidet Innen/Außen ohnehin am Bild.

NEUE KARTE HINZUFÜGEN
──────────────────────
1. Mit debug_ram.py über die neue Grenze laufen.
2. Letzte Position der alten Karte + erste (eingeschwungene!) Position
   der neuen Karte ablesen (NICHT den 1-Frame-Restwert direkt am Sprung).
3. Offset rechnen:  offset_neu = offset_alt + lokal_alt - lokal_neu
   (komponentenweise; bei Nord-Süd-Naht für x, bei Ost-West-Naht für y).
4. Eintrag in MAP_OFFSETS ergänzen.
"""

import numpy as np


class GlobalCoordinateTransform:
    """
    Wandelt lokale (map_bank, map_number, x, y) in normalisierte globale
    Weltkoordinaten um und merkt sich die letzte Außenposition (Carry-Forward).

    Pro Umgebung EINE Instanz halten und in env.reset() reset() aufrufen –
    so bleibt der Carry-Forward-Zustand sauber pro Episode und kollidiert
    nicht zwischen parallelen Trainings-Umgebungen.

    Verwendung:
        gct = GlobalCoordinateTransform()
        gct.reset()                                   # pro Episode
        gx, gy, indoor = gct.features(bank, num, x, y)
    """

    # ── Naht-Offsets (in read_player_position-Einheiten) ─────────────
    # Empirisch aus debug_ram.py Walk-Through, Anker New Bark Town = (0,0).
    # Disassembly-Namen siehe ram_reader.py.
    MAP_OFFSETS = {
        (24, 4): (0,   0),     # NEW_BARK_TOWN  (Anker)
        (24, 3): (-31, 0),     # ROUTE_29       (westlich von New Bark)
        (26, 3): (-52, 0),     # CHERRYGROVE_CITY
        (26, 1): (-47, -28),   # ROUTE_30  (langer Korridor nach Norden)
        (26, 2): (-57, -38),   # ROUTE_31  (Weg nach Violet City)
        # ── Violet City: SCHÄTZUNG (kein exaktes Stitching möglich) ──────
        # Route 31 → Violet läuft durch ein TORHAUS (Warp), nicht über eine
        # Oberwelt-Naht. Beim Warp teleportiert die Koordinate → keine
        # durchgehende Kante zum Anketten wie bei den anderen Karten.
        # Annahme: Torhaus = dünner Ost-West-Durchgang, Türen auf gleicher
        # Höhe → Violet liegt ~westlich von Route 31 auf gleicher Breite.
        # Rechnung: Route31(4,5)=global(-53,-33); Violet-Eingang (21,14) ~5
        #           Kacheln westlich, gleiche Höhe → offset = (-79, -47).
        # Gut genug für Feature-Input + Frontier-Reward; bei Bedarf justieren.
        (10, 5): (-79, -47),   # VIOLET_CITY  (geschätzt, via Torhaus-Warp)
        # ── Route 32 & 36: ECHTE Oberwelt-Nähte zu Violet (debug_ram Walk-Through) ──
        # Beide via offset = offset_Violet + lokal_Violet - lokal_neu, für BEIDE
        # Laufrichtungen geprüft (0-Sprung an der Naht):
        #   Route 32 (südl.): Violet(9,20) ↔ R32(9,2)   → (-79,-47)+(9,20)-(9,2)
        #   Route 36 (westl.): Violet(1,6) ↔ R36(31,6)  → (-79,-47)+(1,6)-(31,6)
        (10, 1): (-79,  -29),  # ROUTE_32  (südlich von Violet)
        (10, 3): (-109, -47),  # ROUTE_36  (westlich von Violet)
        # ── Union Cave: WARP von Route 32 Süd (Höhleneingang, keine Oberwelt-Naht) ──
        # Naht-Formel aufs Warp-Paar: offset = offset_R32 + R32(5,41) - UnionCave(10,3)
        #                                    = (-79,-29) + (5,41) - (10,3) = (-84, 9)
        # → beide Warp-Tiles fallen auf Weltkoord (-74,12) = 0-Sprung am Übergang.
        # Hinweis: gestitchte Höhle → is_indoor wird hier 0 (gewollt: echte Pos + Süd-Gain).
        (3, 29): (-84,    9),  # UNION_CAVE 1F (Eingang von Route 32, weiter nach Süden)
        # ── Route 33 + Azalea City: Höhlenausgang nach Westen (debug_ram Walk-Through) ──
        # Union-Cave-Ausgang (10,17) ↔ Route 33 (7,6) → offset = (-84,9)+(10,17)-(7,6) = (-81,20)
        # Route 33 (1,9) ↔ Azalea (21,9), Oberwelt-Naht → (-81,20)+(1,9)-(21,9) = (-101,20)
        (8, 6): (-81,   20),   # ROUTE_33  (Höhlenausgang → Westen nach Azalea)
        (8, 7): (-101,  20),   # AZALEA_CITY  (Ziel: 2. Arena/Bugsy, von Osten betreten)
        # TODO Route 46 (5,9): ebenfalls Torhaus-Warp → vorerst Frontier (is_indoor)
    }

    # ── Normalisierung auf [0, 1] ────────────────────────────────────
    # global + WORLD_ORIGIN, dann / WORLD_SIZE.
    # Großzügig gewählt damit ganz Johto (und später Kanto) hineinpasst
    # ohne Re-Skalierung. New Bark liegt dadurch bei ~0.5/0.5; der Westen
    # und Norden (negativ) sowie Kanto im Osten (positiv) haben Reserve.
    # Bei Bedarf später enger ziehen für mehr Auflösung.
    WORLD_ORIGIN = 256
    WORLD_SIZE   = 512.0

    def __init__(self):
        self.reset()

    def reset(self):
        """Pro Episode aufrufen – verwirft die Carry-Forward-Historie."""
        self._last_outdoor = (0, 0)   # letzte bekannte Außenposition (global, roh)

    def to_global(self, map_bank, map_number, local_x, local_y):
        """
        Rohe globale (gx, gy) oder None, falls die Karte unbekannt ist
        (Innenraum oder noch nicht gestitchte Frontier-Karte).
        """
        off = self.MAP_OFFSETS.get((map_bank, map_number))
        if off is None:
            return None
        return (off[0] + local_x, off[1] + local_y)

    def features(self, map_bank, map_number, local_x, local_y):
        """
        Liefert (global_x_norm, global_y_norm, is_indoor) ∈ [0, 1]³.

        Oberwelt   → echte normalisierte Weltposition, is_indoor = 0.0
        Innen/Front → Carry-Forward der letzten Außenposition, is_indoor = 1.0
        """
        g = self.to_global(map_bank, map_number, local_x, local_y)
        if g is None:
            is_indoor = 1.0
            g = self._last_outdoor            # nicht nach (0,0) springen
        else:
            is_indoor = 0.0
            self._last_outdoor = g            # für nächsten Innenraum merken

        gx = (g[0] + self.WORLD_ORIGIN) / self.WORLD_SIZE
        gy = (g[1] + self.WORLD_ORIGIN) / self.WORLD_SIZE
        gx = float(np.clip(gx, 0.0, 1.0))
        gy = float(np.clip(gy, 0.0, 1.0))
        return gx, gy, is_indoor

    def global_with_fallback(self, map_bank, map_number, local_x, local_y):
        """
        Rohe globale Position (NICHT normalisiert) – für Distanzberechnungen.
        Bei Innenraum/Frontier: letzte bekannte Außenposition (Carry-Forward),
        ohne den internen Zustand zu verändern.
        """
        g = self.to_global(map_bank, map_number, local_x, local_y)
        return g if g is not None else self._last_outdoor
