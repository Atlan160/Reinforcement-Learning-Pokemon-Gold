# Projekt-Protokoll — Reinforcement Learning Pokémon Gold

**Stand:** 2026-06-24
**Umfang:** Zusammenfassung der Änderungen, Entscheidungen und Diagnosen aus einer längeren Arbeitssitzung.
**Ziel des Projekts:** PPO-Agent (stable-baselines3, PyBoy) spielt deutsches Pokémon Gold (`PGV.gbc`):
New Bark Town → Route 29 → Cherrygrove → Route 30/31 → Violet City → Falkner-Arena (1. Orden) → Route 32 → Union Cave → Route 33 → Azalea.

---

## 0. Die wichtigsten Änderungen auf einen Blick

| # | Änderung | Wirkung |
|---|---|---|
| 1 | **Orden-RAM-Adresse `0xD857 → 0xD57C`** | Kritischer Bugfix: der gesamte Orden-Mechanismus war vorher tot |
| 2 | **Gain-Reward (Pfad-Fortschritt)** | **Entscheidender Durchbruch:** dichtes, richtungs-korrektes Vorwärtssignal *entlang der Route* statt radialer Distanz — zieht die KI gezielt zum Ziel statt in Sackgassen |
| 3 | **Beobachtungsraum 11 → 17 Features** | Geld raus; PP, Team-Level, Ei, 4 Navigations-Features rein |
| 4 | **Navigations-Gedächtnis** (Richtung, map_changed, steps_since) | De-Aliasing gegen Pendeln — billiger als ein LSTM |
| 5 | **Zentrale Reward-Konstanten** | Reward-Tuning an einer Stelle |
| 6 | **Map-Stitching erweitert** | Route 32, Union Cave, Route 33, Azalea + richtungs-korrekte Gains |
| 7 | **Crash-Robustheit** | Worker-Absturz killt nicht mehr das ganze Training; Modell wird gerettet |
| 8 | **Performance +~50 %** | Nur 1× rendern pro Schritt statt 24× (+ Sound aus, Einzel-Graustufen) |

---

## A. Kritischer Bugfix: Orden-Adresse (0xD857 → 0xD57C)

**Problem:** `read_badge_count` las die Johto-Orden bei `0xD857`. Das ist die **Crystal**-Adresse. Dieses ROM nutzt die **Gold/Silber**-WRAM-Karte (passt zu den übrigen verifizierten Adressen wie Party `0xDA22`). In Gold blieb `0xD857` permanent 0.

**Folge:** Der **gesamte Orden-Mechanismus war tot** — `BADGE_REWARD` feuerte nie, das Episode-Ende bei Orden griff nie, `mean_badges` war immer 0, selbst wenn die KI Falkner schlug.

**Fix:** `JOHTO_BADGES = 0xD57C` in `ram_reader.py:194` (Bitfeld, Bit0=Falkner … Bit7=Clair).
**Verifikation:** Mit `debug_ram.py` manuell Falkner besiegt → `0xD57C` springt sauber `0 → 1`, `0xD857` bleibt 0. Eindeutig bestätigt.

→ Nach dem Fix zeigte sich: `mean_badges ≈ 0.2`, und `badges/arena ≈ 0.8` — die KI **gewinnt** Falkner zuverlässig, *wenn* sie die Arena erreicht.

---

## B. Beobachtungsraum (RAM-Features): 11 → 17

Konstante: `config.py → N_RAM_FEATURES`. Aufbau in `ram_reader.py → get_all_ram_features` (RAM-Teil) + `pokemon_env.py → _nav_features` (Navigations-Teil).

**Entwicklung:** 11 (mit Geld) → Geld entfernt (10) → +PP +Team-Gesamtlevel (12) → +Ei (13) → +4 Navigations-Features (17).

**Finaler 17-Feature-Vektor:**

| # | Feature | Normierung |
|---|---|---|
| 1–2 | lokale X / Y | `/255` |
| 3–4 | globale Welt-X / Y (gestitcht) | [0,1] |
| 5 | is_indoor | 0/1 |
| 6 | HP-Ratio | `/maxHP` |
| 7 | Team-Größe | `/6` |
| 8 | **Team-Gesamtlevel** | `/600` |
| 9 | **PP (Mon 1, alle Moves)** | `/200` |
| 10 | Gefangene Pokémon | `/251` |
| 11 | Badges | `/16` |
| 12 | **Ei im Team** | 0/1 |
| 13 | Kampfstatus | 0/1 |
| 14–15 | **Richtung X / Y** (geglättete Bewegung) | [0,1], 0.5 = Stillstand |
| 16 | **map_changed** | 0/1 |
| 17 | **steps_since_map_change** | `/300` |

**Entscheidungsprinzip (durchgehend):** Nur Features aufnehmen, die eine *konkrete Entscheidung* informieren und schwer aus Pixeln ablesbar sind.
- **Geld entfernt:** für das aktuelle Ziel irrelevant (KI kauft nichts). Achtung: jede Feature-Änderung ändert die Netz-Eingabe → **alte Checkpoints inkompatibel** (frischer Lauf nötig).
- **Tile-/Map-Zähler verworfen:** reine Zähler sind „Uhren", nicht handlungsrelevant, und würden eher das Farmen optimieren.
- **PP / Team-Level / Ei / Richtung aufgenommen:** echte Zustands-Bits, die die optimale Aktion ändern.

---

## C. Navigations-Gedächtnis (De-Aliasing gegen Pendeln)

**Problem:** Die (gedächtnislose) Policy sieht an Position P dasselbe, egal ob sie vorwärts oder zurück läuft → an mehrdeutigen Stellen pendelt sie (z. B. NBT ↔ Route 29, bis Timeout). Das ist klassisches **State-Aliasing**.

**Lösung statt LSTM:** drei billige, explizite Features (`pokemon_env.py → _nav_features`):
- **Richtung X/Y** — geglättete (EMA, α=0.3) Bewegungsrichtung. De-aliased „vorwärts vs. zurück". Skala: `<0.5` = negative Achse, `>0.5` = positive, `0.5` = Stillstand. Wert sättigt nach ~3–4 Schritten (= „gehe ich *gerade* konsistent in diese Richtung", kein linearer Zähler).
- **map_changed** — 1.0 im Schritt eines Kartenwechsels (z. B. „grad aus Gebäude raus").
- **steps_since_map_change** — linearer Trödel-/Stuck-Zähler bis `NAV_STEPS_CAP = 300`.

**Hybride Position (Arena-Fix):** Die Richtung wird DRAUSSEN aus **globalen** Koordinaten (stetig an Nähten), DRINNEN aus **lokalen** x/y berechnet (`to_global` → Fallback lokal). So funktioniert die Richtung auch in Arena/Gebäuden, wo die globale Position eingefroren wäre. Der ±1-Clamp fängt den Frame-Wechsel an der Tür ab.

**Warum kein LSTM:** Kurzzeit-Gedächtnis (1–20 Schritte) ist zwar LSTM-Stärke, aber explizite Features lösen genau die zwei benannten Fälle billiger, schneller (kein ~1.5–2× Tempo-Verlust) und kontrollierbar. LSTM lohnt erst bei *open-ended* Gedächtnis. (Langzeit „war ich vor 2000 Schritten im Turm" ist ohnehin LSTM-Schwäche.)

---

## D. Reward-Struktur: zentrale Konstanten

Alle Reward-Größen sind als benannte Modul-Konstanten oben in `pokemon_env.py` gebündelt („ZENTRALE REWARD-KONFIGURATION") → Tuning an einer Stelle, nicht im Code verstreut.

Konstanten (Ausgangswerte; werden vom User getunt):
`NEW_TILE_REWARD`, `PROGRESS_REWARD`, `NEW_MAP_REWARD`, `DMG_REWARD`, `XP_REWARD`, `LEVEL_UP_REWARD`, `TRAINER_WIN_REWARD`, `NEW_SPECIES_REWARD`, `NEW_PARTY_REWARD`, `CATCH_REWARD`, `HEAL_REWARD`, `PP_REFILL_REWARD`, `BADGE_REWARD`, `TRAP_PENALTY`, `SCROLL_PENALTY`, `FLEE_PENALTY`, `HP_LOSS_PENALTY`, `TIME_PENALTY`, `FLEE_STUCK_BONUS`. (Ortsgebundene Meilensteine bleiben separat im `MILESTONE_BONUS`-Dict.)

**Grind-Sättigung (Decay-Faktor) — sauberer Anti-Camping-Mechanismus:**
`saturation_decay(K, farmed) = min(1, K / (1 + farmed))` (`pokemon_env.py`) skaliert XP- und Schadens-Reward herunter, je mehr die KI **auf der aktuellen Karte** schon gefarmt hat:
- **Gnadenzone:** die ersten ~K Einheiten zahlen vollen Reward (Faktor 1.0), danach **hyperbolischer Abfall** (Halbwert bei ~2K, erreicht nie exakt 0).
- `farmed` = bereits auf dieser Karte gefarmte XP (kanonisches Grind-Maß), pro Karte zurückgesetzt. Greift gemeinsam für **XP-** und **Schadens-Reward**.
- **Pro-Karte abstimmbar:** `GRIND_SATURATION_DEFAULT` (~120) + `GRIND_SATURATION_PER_MAP` (z. B. Route 30/31/32 höher, Violet-Arena ~9990 = praktisch unbegrenzt, damit der Falkner-Kampf immer voll zählt).

**Warum gut:** Etwas Leveln ist erlaubt/nötig, aber **Dauer-Camping auf einer Karte lohnt nicht** → die KI wird sanft weitergeschoben, ohne dass man Kämpfe hart verbieten muss. Glatt, tunebar, ergänzt Gain-Reward + Pro-Episode-Logik.

**Trap-Maps — gegen Verlaufen in nutzlosen Innenräumen:**
Manche Innenräume (Gebäude, Alph-Ruinen, Torhäuser) bringen der KI nichts — sie verläuft sich dort nur. Diese stehen in `_trap_maps` (`pokemon_env.py → reset()`):
- Beim **ersten** Betreten pro Episode gibt's statt des Map-Entdeckungs-Bonus (`+NEW_MAP_REWARD`) eine **kleine Strafe** (`-TRAP_PENALTY`).
- Trap-Maps sind zusätzlich in `_cleared_maps` → **kein** XP-/Tile-Reward-Farming drin.
- **Beispiele:** New-Bark-/Cherrygrove-/Violet-Gebäude, Pokémart, Pokécenter-OG, Mr.-Pokémons-Haus, Alph-Ruinen-Innenräume `(3,22/24/27/28)`, Alph-Torhäuser `(10,12/16)`.
- **Bewusst NICHT dabei:** Arena `(10,7)`, Pokécenter-EG `(26,5)` — dorthin *soll* sie (kämpfen / heilen lernen). *(Knofensa-Turm `(3,1)` war hier, ist seit dieser Session **Trap** — die KI fand nicht mehr raus; Details in Nachtrag T.)*

**Warum gut:** Hält die KI auf der produktiven Route, ohne ihr die Bewegungsfreiheit zu nehmen — sie lernt schnell, nutzlose Türen zu meiden. (In dieser Session um Violet-Gebäude + Alph-Ruinen erweitert.)

**Kampf-Verhalten — Flucht-Strafe, Flucht-Bonus & Stuck-Detektor:**
Ein **Kein-Schaden-Zähler** `_battle_steps_no_damage` (`pokemon_env.py`) misst **aufeinanderfolgende** Kampf-Schritte ohne Schaden: bei *jedem* Treffer → 0, sonst +1. Er erkennt Festfahren/Geflatter *mitten* im Kampf und treibt drei Effekte:
- **Flucht aus *kämpfbarem* Kampf → Strafe** (`-FLEE_PENALTY`, 3.0): nicht vor gewinnbaren Kämpfen weglaufen.
- **Flucht aus *aussichtslosem* Kampf → Bonus** (`+FLEE_STUCK_BONUS`, 0.7): ab **≥100** schadenlosen Schritten (z. B. leere Angriffs-PP → kann nicht treffen) ist Fliehen *richtig* → belohnt statt bestraft.
- **Bag-/Menü-Geflatter bremsen** (`-SCROLL_PENALTY`, 0.003/Schritt): ab **≥50** schadenlosen Schritten im Kampf wird zielloses Scrollen teuer. In echten Kämpfen feuert das nie (ein Treffer setzt den Zähler auf 0).
- **Fehlalarm-Schutz:** `_enemy_defeated_this_battle` verhindert eine falsche Flucht-Strafe, wenn der Gegner stirbt *und* der Kampf in denselben 24 Frames endet.

(Der Schadens-Reward `DMG_REWARD` selbst läuft durch denselben **Decay-Faktor** wie XP — gleiche Per-Karte-Sättigung.)

**Warum gut:** Die KI lernt, gewinnbare Kämpfe auszufechten, aussichtslose (leere PP) sauber zu *verlassen* statt sich festzubeißen, und nicht im Menü zu trödeln — ohne dass man Kampf-/Flucht-Aktionen hart verbieten muss.

**Diskutierte Reward-Design-Erkenntnisse:**
- Pro-Episode wieder kassierbare Belohnungen (Meilensteine, +8 pro Karte, Tile-Reward, Kämpfe/Level/Fänge) zahlen die KI fürs **Im-Kreis-Laufen** → Oszillation, kein stabiler Anker.
- Der **Orden beendet die Episode** → er muss eine *ganze* Farm-Episode schlagen (Richtung +300–500 statt 50), sonst ist „gewinnen" netto ein Verlust.
- `NEW_MAP_REWARD` entfernen schwächt aber den „raus aus NBT"-Sog → Vorsicht (siehe Trainings-Analyse).

---

## E. Gain-Reward (Pfad-Fortschritt) & Map-Stitching

### E.1 Der Gain-Reward — der entscheidende Durchbruch

**Kernidee:** Statt **radialer Distanz** zum Start (die *jede* Bewegung weg vom Start belohnt — auch in Sackgassen) misst der Gain-Reward den Fortschritt **entlang der intendierten Route** (`route_progress`) und belohnt nur die **Zunahme** (den „Gain") pro Schritt. Ergebnis: ein **dichtes, richtungs-korrektes Vorwärtssignal** — die KI wird gezielt die Route entlanggezogen (West → Nord → NW → Süd → West), Rückwärts/Seitwärts gibt **keinen** Reward.

**Warum das *der* Durchbruch war:** Radiale Distanz zog die KI in jede Entfernung (auch Sackgassen wie die Westküste). Erst der pfad-gerichtete Gain brachte sie *zuverlässig* von New Bark bis Route 31/Violet — die `mean_dist_from_start`-Kurve stieg sauber, statt im Kreis zu laufen. Er ist das Fundament, auf dem alles Spätere (Stitching, Süd-/West-Gains, Curriculum) aufbaut.

**`route_progress`** (`pokemon_env.py`): pro Streckenabschnitt („Leg") eine richtungs-korrekte Formel auf den Weltkoordinaten, z. B. Route 29 `10 - gx` (West), Route 30/31 `46 - gy` (Nord), Violet `(46-gy)+0.6·(-58-gx)` (NW). Innenräume/ungemappt → `None` (kein Richtungs-Reward dort).

**`_stable_prog`-Mechanik (Tod-/Teleport-robust *und* nicht ausnutzbar)** — die eigentliche Kunst:
- **Reward** = `gain × PROGRESS_REWARD` für `0 < gain ≤ 2.1` (kontinuierliche Vorwärtsbewegung); die Frontier `_max_dist_from_start` zieht mit.
- **Großer Sprung (>2 ggü. letztem *stabilen* Wert)** = Tod / Teleport / Warp → Frontier wird neu gesetzt (auch nach unten), **KEIN** Reward → danach zahlt das Wiederhochlaufen **VOLL** (dichtes Signal nach Respawn).
- **Kleiner Sprung (Tür/Naht ≤2)** = kontinuierlich → Frontier bleibt monoton → **kein Hin-und-her-Farm** an Türen/Kanten.
- **2-Schritt-Stale-Filter** (zwei nahe Messungen in Folge) gegen Glitch-Frames direkt nach dem Warp.

→ Robuster als ein einmaliger „Check nach dem Warp": fängt den Respawn auch dann, wenn der erste Frame nach dem Tod ein Restwert nahe der alten Position ist.

**Die zwei Seiten der `>2`-Schwelle — eine Regel, zwei Probleme:**

- **Reset bei Tod (bewusst eingeführt):** Stirbt die KI (Team ohnmächtig → Blackout → Respawn im Pokécenter), teleportiert die Position. Der *große* Sprung löst die Re-Baseline aus → die Frontier wird auf die **Respawn-Position gesetzt** („Distanz-Reset bei Tod"), das Wiederhochlaufen zahlt **vollen** Reward. Ohne das blieb die Distanz nach dem Tod kleben (Bug *„bleibt bei 19 stehen"*) bzw. der Rückweg gab nichts mehr — mit dem Reset bekommt die KI nach *jedem* Tod sofort wieder ein dichtes Vorwärtssignal.
- **Door-Farm-Exploit (verworfen):** Eine frühere Variante (`_after_warp` + `REVISIT_REWARD_FACTOR`) setzte die Frontier bei **jedem** Warp neu — also auch an **Türen / Höhlen- / Arena-Eingängen**. Folge: Die KI lief eine Tür **rein und raus**, um den Fortschritts-Reward immer wieder zu kassieren (~0.75 pro Zyklus) → eskalierte auf `mean_reward ≈ 4000`, `ep_length ≈ 25000`. Behoben durch das `_stable_prog`-Design.

→ Genau hier liegt die Eleganz: **großer** Sprung (echter Tod/Teleport) **→ Reset**; **kleiner** Sprung (Tür/Naht ≤2) **→ kontinuierlich, kein Farm**. Dieselbe Schwelle löst „Distanz klebt nach Tod" *und* „Tür-rein-raus-Farm". (Verworfene Alternativen ausführlich in `memory/progress_reward_monotonic.md`.)

**In dieser Session erweitert:** neue richtungs-korrekte Legs für Route 32 (Süd), Union Cave (Süd), Route 33 & Azalea (West) — plus der Warp-Dead-Zone-Fix (siehe E.3).

### E.2 Weltkoordinaten / gestitchte Karten

Datei: `global_coords.py` (`MAP_OFFSETS`) + `pokemon_env.py` (`route_progress`, `ROUTE_ORDER`).
Anker: New Bark Town = (0,0). `global = offset + lokal`.

**Neu gestitchte Karten & richtungs-korrekte Gains (Legs):**

| Karte | map_key | Offset | Progress-Leg (Gain-Richtung) |
|---|---|---|---|
| Route 32 | (10,1) | (-79, -29) | Süd: `109.2 + gy` |
| Union Cave 1F | (3,29) | (-84, 9) | Süd: `116.0 + gy` |
| Route 33 | (8,6) | (-81, 20) | West: `61.2 - gx` |
| Azalea City | (8,7) | (-101, 20) | West: `61.2 - gx` (provisorisch, bis Bugsy-Arena lokalisiert) |

Die durchgehende, knickfreie Fortschritts-Achse dreht: **West** (NBT/R29) → **Nord** (Cherrygrove/R30/31) → **NW** (Violet) → **Süd** (R32/Union Cave) → **West** (R33/Azalea).

### E.3 Wichtige Lehre — Warp ≠ Naht (Dead-Zone-Fix)

- **Glatte Oberwelt-Naht** (durchlaufbare Kante): Progress *stetig* halten (0-Sprung), keine Re-Baseline.
- **Warp** (Höhle/Torhaus): NICHT auf 0-Sprung zwingen. Wenn die Vor-Karte am Eingang *vorbeireicht*, überschießt die Frontier den Eingang → **Dead-Zone** (kein Reward trotz Vorwärtslauf). Lösung: die neue Leg **deutlich über das Maximum der Vor-Karte** starten, damit der Warp-Sprung die Teleport-Re-Baseline auslöst.
- **Konkret (Union Cave Dead-Zone-Fix):** Route 32 reicht südlich am Höhleneingang vorbei (Frontier ~124), die Höhle startete bei ~121 → Frontier klebte. Konstante `109.2 → 116` (Eingang ~128) → Re-Baseline feuert → Süden zahlt voll. Per Simulation verifiziert.

---

## F. Robustheit / Crash-Handling

**Symptom:** `BrokenPipeError [WinError 109]` / `EOFError` → ein Worker-Subprozess starb, riss das ganze Training (alle 20 Envs) mit.

**Zwei-Schichten-Absicherung:**
- **Schicht 1 (`pokemon_env.py`):** `step()`/`reset()` sind dünne Wrapper um `_step_impl`/`_reset_impl`. Eine Python-Exception im Worker wird abgefangen, nach `crash_logs/` protokolliert, die Episode sauber beendet bzw. PyBoy hart neugestartet → Worker überlebt, Pipe bleibt heil.
- **Schicht 2 (`train.py`):** `model.learn()` fängt `EOFError`/`BrokenPipeError` ab → **Modell wird gespeichert** statt Totalverlust.
- **`faulthandler`** aktiviert → native Crashes (Segfault) drucken den Python-Stack.

**Diagnose der echten Ursache:** `Windows fatal exception: access violation` in **`pokemon_env.py:677` = `self.pyboy.tick()`** → **nativer PyBoy-Bug** (kein Code-Fehler). Über Milliarden tick-Aufrufe statistisch unvermeidlich. PyBoy 2.7.0 ist bereits aktuell → kein Upgrade-Fix. RAM (68 GB) und VRAM (12 GB) als Ursache ausgeschlossen.

**Nicht umgesetzt (bewusst):** Auto-Restart-Schleife (würde das Training nach Crash selbst fortsetzen) — vom User vorerst verworfen.

---

## G. Performance-Optimierungen

Im „rechne/render nur, was du brauchst"-Geist (`pokemon_env.py → _step_impl` / `_get_screen`):
- **+~50 %:** `self.pyboy.tick(FRAMES_PER_ACTION, True)` statt 24× `tick()` → emuliert dieselben Frames, **rendert nur das letzte Bild** statt jedes. Verhaltensidentisch (die KI sieht eh nur das letzte Bild).
- **~3 %:** `sound=False` im tick → ungenutzte APU-Emulation sparen.
- **klein:** Graustufen in *einem* Durchgang (`cv2.COLOR_RGBA2GRAY` statt RGBA→RGB→GRAY), pixelidentisch.

---

## H. Ei-Erkennung (Togepi-Ei als Gate)

Spiel-Gate: nach Falkner gibt Elms Assistent das Togepi-Ei; erst damit geht's auf Route 32 weiter.
**Erkennung:** Ein Ei belegt einen Team-Slot mit Spezies-Marker **`0xFD` (253, „EGG")** in der Party-Liste (`0xDA23`). `read_has_egg` = `0xFD in read_party_species` (`ram_reader.py`). Empirisch bestätigt (beim Abholen erscheint `0xFD`, kurzer 1-Frame-Übergang über 175=Togepi). Als Feature #12 eingebaut.

---

## I. TensorBoard-Crash (separat, harmlos)

`access violation` in numpy 2.2.6 / TensorBoard beim Garbage-Collecting — **nichts mit dem Training zu tun** (eigener Viewer-Prozess, liest nur die Logs). Sporadisch (Race: Reloader liest Event-Dateien, während Training schreibt).
**Abhilfe:** `tensorboard --logdir ./tensorboard_logs/ --load_fast=true --reload_interval 30` (Rust-Datenlader umgeht den crashenden numpy-Pfad).

---

## J. Trainings-Analyse & Diagnose

Aus den TensorBoard-Kurven mehrerer Läufe:
- **Kampf/Aufbau steigt sauber:** caught, party_level (→22), XP, trainers_defeated, reward.
- **`mean_badges ≈ 0.2`**, `arena_visited ≈ 0.25` → **die KI gewinnt Falkner zu ~80 %, WENN sie die Arena erreicht.** Kämpfen ist *nicht* das Problem.
- **Engpass = lebend bis zur Arena kommen** (~25 %). Indizien: hohe Kampf-Metriken (überkämpft), Episodenlänge ~8000 + `pc_heals ≈ 0` → viele Episoden enden per **Tod** unterwegs.
- **Oszillation** (limit cycle) bleibt → farmbare Wiederhol-Rewards ohne dominanten terminalen Anker. Niedrigere LR (5e-5) + γ 0.997 haben es beruhigt, nicht beseitigt.
- **Start-Dithering NBT ↔ Route 29:** State-Aliasing (nicht zu schwaches Reward, da sie Route 29 *erreicht*) → adressiert durch das Navigations-Gedächtnis (Abschnitt C).

**Empfohlene Hebel:** Kampf-Grind dämpfen (v. a. Fang-Reward), `BADGE_REWARD` dominant machen, Curriculum stärker Richtung Gym gewichten.

---

## K. Konfiguration (vom User gesetzt, `config.py`)

Zentrale Hyperparameter (vom User **laufend getunt** — Momentaufnahme bei Protokoll-Erstellung): `LEARNING_RATE = 1.5E-4`, `N_STEPS = 4096`, `BATCH_SIZE = 1024`, `N_EPOCHS = 4`, `GAMMA = 0.997`, `GAE_LAMBDA = 0.95`, `CLIP_RANGE = 0.2`, `ENT_COEF = 0.04`, `VF_COEF = 0.5`, `N_ENVS = 20`, `TOTAL_TIMESTEPS = 25_000_000`, `FRAMES_PER_ACTION = 24`, `MAX_STEPS_PER_EPISODE = 40960`. (Im Verlauf schwankte LR zwischen 1.5e-4 … 5e-5, ENT_COEF zwischen 0.02 … 0.04, BATCH_SIZE 512 … 2048.)
Savestate-Curriculum mehrfach umgestellt (mal auf wenige nahe Starts reduziert für Transfer Learning, mal volle Bandbreite Route 29/30/31 + Gym + Route 32). **Hinweis:** Beobachtungsraum-Änderungen (→17) brechen das Transfer Learning → frischer Lauf nötig.

**PPO-Grundlagen (besprochen):** `n_steps × n_envs` = Rollout-Puffer (gesammelte Datenmenge); `batch_size` = Minibatch-Größe *innerhalb* davon (muss `n_steps × n_envs` glatt teilen). Bei großer Batchsize LR nicht zu klein wählen.

---

## L. Diskussionen ohne Code-Änderung

- **N_steps vs. batch_size** (PPO-Datenfluss erklärt).
- **Badge-Termination-Logging:** bestätigt, dass der Callback den Endwert korrekt einsammelt (speichert jeden Schritt, hängt beim erkannten Reset den gespeicherten Wert an).
- **LSTM-Abwägung** (zweimal): für die konkreten Fälle explizite Features statt Recurrent-Netz.

---

## M. Geänderte Dateien (Übersicht)

| Datei | Wesentliche Änderungen |
|---|---|
| `ram_reader.py` | Orden-Adresse `0xD57C`; `read_has_egg`; PP/Level/Ei in Feature-Vektor; Geld raus |
| `pokemon_env.py` | Zentrale Reward-Konstanten; Crash-Wrapper + faulthandler; tick-/Sound-/Graustufen-Speedup; `route_progress`-Legs (R32, Union Cave, R33, Azalea) + Dead-Zone-Fix; `_nav_features` (hybrid) |
| `global_coords.py` | `MAP_OFFSETS`: Union Cave, Route 33, Azalea |
| `config.py` | `N_RAM_FEATURES = 17`; Hyperparameter & Savestate-Curriculum (vom User) |
| `train.py` | Crash-Sicherheitsnetz (EOFError → Modell speichern); `mean_azalea_visited`-Logging (vom User) |
| `debug_ram.py` | Orden-Watcher (D57C/D857), Ei-/Party-Spezies-Watcher |
| `.gitignore` | `crash_logs/` ergänzt |

---

## N. Gedächtnisdateien (`memory/`)

- `ram-map-is-gold-silver-not-crystal.md` — ROM nutzt Gold/Silber-WRAM, nicht Crystal (Orden `0xD57C`).
- `proactive-performance-optimization.md` — billige Speedups proaktiv vorschlagen.
- `progress_reward_monotonic.md` — `_stable_prog`-Design + Warp-≠-Naht-Regel (Dead-Zone).
- `project_pokemon_gold_rl.md` — Projektstruktur/Ziele.

---

## O. Offene Punkte / für den nächsten Lauf beobachten

1. **`mean_badges`** — steigt sie über 0.2? (Erfolgsmaßstab seit Adress-Fix.)
2. **Pendeln NBT ↔ Route 29** — verschwindet es durch die Navigations-Features? (Wenn ja: war Aliasing. Wenn nein: doch zu schwaches Start-Reward.)
3. **`mean_dist_from_start`** in den ersten ~1–2 Mio. Schritten — kommt sie aus dem Startbereich raus?
4. **Union Cave** — bekommt sie jetzt Süd-Reward (Dead-Zone-Fix)?
5. **Azalea-Leg** ist provisorisch reiner West-Gain — bei Bedarf Bugsy-Arena lokalisieren und Nudge/Meilenstein ergänzen.
6. Offene Adress-Recherchen künftig mit **Gold/Silber**-Quellen (nicht Crystal).

---

# NACHTRAG — Folge-Iterationen (Post-Gym-Progression, Ei-Mechanik, Per-Gruppe-Logging)

*Alles seit der letzten Doku-Aktualisierung.*

## P. Progression ÜBER die Arena hinaus freigeschaltet

- **Der Orden beendet die Episode NICHT mehr** — die Abbruch-Bedingung `if read_badge_count>=1: return True` (`pokemon_env.py → _check_terminated`) ist **auskommentiert**. Folge: Die KI fließt nach dem Orden **weiter** (Richtung Route 32 / Union Cave), statt die Episode dort zu beenden. *Kritischer Nebeneffekt, der dadurch vermieden wird:* Post-Gym-Savestates (Route 32, die den Orden schon haben) würden sonst auf **Schritt 1** sofort terminieren. Der Orden wirkt jetzt als Meilenstein, nicht als Endpunkt.
- **Eintritts-Meilensteine erweitert** (`MILESTONE_BONUS`): `(3,29)` Union Cave **+20**, `(8,7)` Azalea **+20** — Sog über die **Warp-Schwellen** (die Gains zahlen an Warps wegen der Re-Baseline nicht, siehe E.3).

## Q. Ei-Mechanik (erweitert/korrigiert Abschnitt H)

Das Togepi-Ei aus dem Violet-Pokécenter ist ein echtes **Spiel-Gate**: ohne Ei kommt man auf Route 32 nicht weiter. Drei Änderungen:

- **Ei-LINIE statt nur Ei** (`read_has_egg`, `ram_reader.py`): zählt jetzt `0xFD` (Ei) **ODER** `175` (geschlüpftes Togepi). → Robust gegen das Schlüpfen. Behebt auf einen Schlag: die **Metrik** `mean_has_egg` (fiel beim Schlüpfen fälschlich auf 0 → unterzählte), die **Observation #12** (jetzt = sauberes „durch das Gate"-Signal) **und** macht den Ei-Reward bombenfest. (175=Togepi empirisch bestätigt — im Ei-Abhol-Trace tauchte 175 kurz vor `0xFD` auf.)
- **Ei-Reward-Exploit gefixt:** `if has_egg and not _prev_egg_state` → feuert nur bei **0→1** (Ei geholt). Vorher feuerte `_prev != has_egg` auch beim **Schlüpfen** (1→0) → nochmal **+150** für etwas Unerwünschtes (die KI nutzte das aus).
- **`PC_ENTRY_REWARD` (+30):** erstes Betreten des Violet-Pokécenters `(10,10)` pro Episode → **Sog in den Ei-„Entdeckungsraum"** (der NPC sitzt dort). Bewusst ein **eigener Flag** (`_pc_egg_entered`), NICHT `MILESTONE_BONUS`: `(10,10)` würde via `ROUTE_ORDER` für Gym-/Route32-Starts vor-markiert → ein Milestone würde dort *nie* feuern — gerade für die Starts, die zum PC *zurück* müssen, um das Ei zu holen.

**Die Ei-Entdeckungs-Kette** (gegen das „seltener NPC-Trigger"-Discovery-Problem): Savestate *am NPC* (`Pokecenter_at_assistant`, **Entdeckung**) → `PC_ENTRY_REWARD` (**Sog** in den Raum, auch für weite Starts) → `EGG_REWARD = 150` (**Verstärkung**). Wichtige Einsicht: `EGG_REWARD` war nie zu klein — das Problem ist rein die **Entdeckung** des seltenen Triggers, nicht der Reward.

## R. Per-Savestate-Gruppe-Logging (start / middle / end)

- **`SAVESTATE_GROUPS`** (`config.py`): ordnet jeden Savestate per **Datei-Namen-Teilstring** einer Gruppe zu — NICHT per Index. Robust dagegen, dass Aus-/Einkommentieren die Indizes verschiebt (genau dieser Bug trat mit dem fehlenden `PC_before_egg` auf: 15 gelistet, aber nur 14 im Pool → alle Indizes danach verschoben).
- **Alle 17 Metriken pro Gruppe** → eigene TensorBoard-Sektionen `start/`, `middle/`, `end/` (= 51 Kurven), zusätzlich zu den aggregierten `pokemon/…`.
- **Neue aggregierte Metriken:** `mean_has_egg`, `mean_union_cave_visited`.

## S. Diagnose-Erkenntnis: Bimodalität

- Die aggregierten Mittelwerte **mitteln zwei Regime** zusammen (Episode schafft *enorm viel* vs. *gar nichts*) und sind stark **savestate-getrieben** (Gym-nahe Starts gewinnen fast immer, weite Starts scheitern). Der Mittelwert repräsentiert *keinen* der Modi → erst das Per-Gruppe-Logging macht es sichtbar.
- Konkret bestätigt: **Ei-Rate ≈ Union-Cave-Rate (~20 %)** → wer das Ei hat, betritt fast immer die Höhle; der Engpass ist *rein* das Ei-Abholen, nicht das Weiterkommen danach.
- **Kaputte Route-29-Savestates** waren ein Teil des Fehl-Modus; nach Reparatur + LR/Entropie runter stieg `mean_badges` von ~0.2 auf **~0.6**. (Niedriger Reward ≠ schlechter: ein früherer Lauf hatte hohen Reward durch *Farmen* bei ~0 Badges — weniger Farmen + mehr Orden ist die bessere Richtung.)

## T. Knofensa-Turm → Trap-Map (Umkehr einer früheren Entscheidung)

- **Änderung:** `(3, 1)` (Turm-Basis) steht jetzt in `_trap_maps` (`pokemon_env.py → reset()`). Beim Betreten also `-TRAP_PENALTY` statt Map-Bonus, und der Turm ist automatisch in `_cleared_maps` → **kein** XP-/Tile-Farming drin.
- **Warum:** Die KI fand aus dem **mehrstöckigen** Turm (Leitern zwischen den Etagen) **nicht mehr heraus** und verbrannte dort ganze Episoden. Für die gedächtnislose Policy ist das Rausfinden zu schwer → der Turm war netto ein **Zeit-/Reward-Grab**. Ihn aktiv zu **bestrafen** (gar nicht reingehen) schlägt das frühere „Reinlocken".
- **Kehrt um:** Abschnitt D listete den Turm noch als *bewusst KEINE* Trap („dorthin *soll* sie"). Die Turm-**Meilensteine** `(3,1)/(3,2)/(3,3)` in `MILESTONE_BONUS` waren schon vorher auskommentiert (kein Sog nach oben mehr) — dieser Schritt geht weiter: von „neutral" zu „aktiv meiden".
- **Nur die Basis `(3,1)` nötig:** Die KI muss durch die Basis, um zu den Etagen `(3,2)/(3,3)` zu kommen; die Basis-Strafe hält sie vom ganzen Turm ab. (Der `GRIND_SATURATION_PER_MAP`-Eintrag `(3,1):300` ist damit gegenstandslos — der Turm ist ohnehin `cleared` —, schadet aber nicht.)
- **Code-Kommentare nachgezogen:** die zwei nun widersprüchlichen Kommentare bei `_trap_maps` („Turm … bewusst NICHT dabei" / „bleibt Meilenstein, kein Trap!") wurden korrigiert.

## Neue offene Punkte

1. **`middle/mean_has_egg`** — holt die KI das Ei aus dem Gym-Bereich ab? (Erfolgssignal der Ei-Kette.)
2. **`start/…` & `middle/…` vs. `end/…`** — kommt der Erfolg aus echter **Navigation** (start/middle steigen) oder nur von den „geschenkten" end-Starts (post-egg)?
3. **Curriculum-Savestates in Union Cave + Route 33** fehlen noch (würden diese Segmente direkt trainieren; die Gains + Eintritts-Meilensteine sind da).
4. **Azalea-Leg** weiterhin provisorisch (reiner West-Gain) — Bugsy-Arena lokalisieren für Nudge/Meilenstein.

## Dateien in diesem Nachtrag

| Datei | Änderung |
|---|---|
| `ram_reader.py` | `read_has_egg` → Ei-Linie (`0xFD` oder `175`) |
| `pokemon_env.py` | Ei-Reward nur 0→1; `PC_ENTRY_REWARD` + `_pc_egg_entered`-Flag; Badge-Termination auskommentiert; Union-Cave/Azalea-Milestones; `_savestate_group()` + `_episode_group`; `_get_info`-Keys `has_egg`/`union_cave_visited`/`savestate_group`; **Knofensa-Turm `(3,1)` → Trap-Map** (T) |
| `config.py` | `SAVESTATE_GROUPS` (namens-basiert) |
| `train.py` | Per-Gruppe-Logging (start/middle/end × 17 Metriken); aggregiert `mean_has_egg` + `mean_union_cave_visited` |
