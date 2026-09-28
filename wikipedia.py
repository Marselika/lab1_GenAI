"""
Extrage articole diversificate de pe Wikipedia RO, le curata
si le salveaza intr-un fisier .txt pentru antrenare LSTM char-level.
"""

import requests
import re
import time
import random
import json
import os

API_URL = "https://ro.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "CorpusBuilder/1.0 (educational LSTM project; contact: exemplu@email.com)"}

CATEGORII = [
    # Literatură și lingvistică
    "Literatură română",
    "Literatură universală",
    "Poezie",
    "Roman",
    "Lingvistică",
    "Scriitori români",

    # Istorie
    "Istoria României",
    "Istorie universală",
    "Istoria antichității",
    "Evul Mediu",
    "Al Doilea Război Mondial",
    "Istoria artei",

    # Filosofie și religie
    "Filosofie",
    "Filosofi",
    "Religie",
    "Creștinism",
    "Mitologie",
    "Mitologie greacă",

    # Științe naturale
    "Astronomie",
    "Biologie",
    "Zoologie",
    "Botanică",
    "Fizică",
    "Chimie",
    "Geologie",
    "Ecologie",
    "Medicină",
    "Anatomie",

    # Geografie
    "Geografia României",
    "Geografie",
    "Țări din Europa",
    "Orașe din România",
    "Munți",
    "Fluvii",

    # Artă și cultură
    "Muzică",
    "Artă",
    "Pictură",
    "Cinematografie",
    "Teatru",
    "Arhitectură",
    "Dans",
    "Fotografie",

    # Societate
    "Sport",
    "Fotbal",
    "Politică",
    "Economie",
    "Drept",
    "Sociologie",
    "Psihologie",
    "Educație",
    "Pedagogie",

    # Tehnologie și științe exacte
    "Tehnologie",
    "Informatică",
    "Matematică",
    "Inginerie",
    "Robotică",
    "Internet",

    # Diverse cu conținut narativ bogat
    "Mitologie română",
    "Folclor românesc",
    "Gastronomie",
    "Bucătăria românească",
    "Agricultură",
    "Meteorologie",
]

ARTICOLE_PE_CATEGORIE = 40
MIN_CARACTERE = 1500
MAX_CARACTERE = 15000
CERERE_PAUZA = 1.0
MAX_INCERCARI = 5
BATCH_SIZE = 15

FISIER_CHECKPOINT = "checkpoint_continuturi.json"
FISIER_CORPUS = "corpus_ro.txt"

session = requests.Session()
session.headers.update(HEADERS)


# ---------------------------------------------------------------------
# Retea: cereri cu retry si backoff
# ---------------------------------------------------------------------

def cerere_cu_retry(params, max_incercari=MAX_INCERCARI):
    """
    Face un request GET cu retry si exponential backoff.
    Respecta header-ul Retry-After daca serverul il trimite.
    Returneaza JSON-ul raspunsului sau None daca a esuat definitiv.
    """
    asteptare = 2

    for incercare in range(1, max_incercari + 1):
        try:
            r = session.get(API_URL, params=params, timeout=20)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                pauza = int(retry_after) if retry_after else asteptare
                print(f"    [429] Prea multe cereri. Astept {pauza}s "
                      f"(incercarea {incercare}/{max_incercari})...")
                time.sleep(pauza)
                asteptare *= 2
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            print(f"    [!] Eroare de retea: {e}. Reincerc in {asteptare}s...")
            time.sleep(asteptare)
            asteptare *= 2

    print(f"    [X] Am renuntat dupa {max_incercari} incercari esuate.")
    return None


def obtine_titluri_din_categorie(categorie, limita=40):
    """Ia titlurile de pagini dintr-o categorie Wikipedia (nu subcategorii)."""
    titluri = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Categorie:{categorie}",
        "cmlimit": limita,
        "cmtype": "page",
        "cmnamespace": 0,
        "format": "json",
    }
    data = cerere_cu_retry(params)
    if data:
        membri = data.get("query", {}).get("categorymembers", [])
        titluri = [m["title"] for m in membri]
    time.sleep(CERERE_PAUZA)
    return titluri


def obtine_continut_articole(titluri, continuturi_existente=None):
    """
    Ia continutul text pentru o lista de titluri, in batch-uri.
    Salveaza checkpoint dupa fiecare batch reusit, ca sa nu pierzi
    progresul daca scriptul e intrerupt.
    """
    rezultate = continuturi_existente or {}
    titluri_ramase = [t for t in titluri if t not in rezultate]

    if not titluri_ramase:
        print("  Toate articolele sunt deja in checkpoint.")
        return rezultate

    print(f"  {len(titluri_ramase)} articole de extras (restul sunt deja in checkpoint)")

    for i in range(0, len(titluri_ramase), BATCH_SIZE):
        batch = titluri_ramase[i:i + BATCH_SIZE]
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "titles": "|".join(batch),
            "format": "json",
            "redirects": 1,
        }
        data = cerere_cu_retry(params)
        if data:
            pagini = data.get("query", {}).get("pages", {})
            for pagina in pagini.values():
                titlu = pagina.get("title", "")
                text = pagina.get("extract", "")
                if titlu and text:
                    rezultate[titlu] = text

            # checkpoint dupa fiecare batch reusit
            salveaza_checkpoint(rezultate)

        progres = min(i + BATCH_SIZE, len(titluri_ramase))
        print(f"    Progres: {progres}/{len(titluri_ramase)} articole noi procesate "
              f"({len(rezultate)} total in checkpoint)")
        time.sleep(CERERE_PAUZA)

    return rezultate


# ---------------------------------------------------------------------
# Checkpoint (salvare/incarcare progres, ca sa nu pierzi munca la 429)
# ---------------------------------------------------------------------

def salveaza_checkpoint(continuturi):
    with open(FISIER_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(continuturi, f, ensure_ascii=False)


def incarca_checkpoint():
    if os.path.exists(FISIER_CHECKPOINT):
        with open(FISIER_CHECKPOINT, "r", encoding="utf-8") as f:
            date = json.load(f)
        print(f"[i] Checkpoint gasit: {len(date)} articole deja extrase anterior.")
        return date
    return {}


# ---------------------------------------------------------------------
# Curatare si filtrare calitate
# ---------------------------------------------------------------------

def curata_text(text):
    """
    Curata textul brut extras de la Wikipedia:
    - elimina sectiuni non-narative (Note, Bibliografie, Legaturi externe etc.)
    - elimina note de tip [1], [citare necesara]
    - elimina linii-titlu orfane (scurte, fara punctuatie finala)
    - normalizeaza spatii si linii goale
    """
    sectiuni_de_taiat = [
        "Note", "Referințe", "Bibliografie", "Legături externe",
        "Vezi și", "Note și referințe", "Surse",
    ]
    for sectiune in sectiuni_de_taiat:
        pattern = rf"\n{re.escape(sectiune)}\n"
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[[^\]]*citare[^\]]*\]", "", text, flags=re.IGNORECASE)

    linii = text.split("\n")
    linii_curate = []
    for linie in linii:
        linie = linie.strip()
        if not linie:
            continue
        if len(linie) < 25 and not linie.endswith((".", "!", "?", ":")):
            continue
        linii_curate.append(linie)

    text = "\n".join(linii_curate)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()


def articol_e_de_calitate(text):
    if len(text) < MIN_CARACTERE:
        return False
    if len(text) > MAX_CARACTERE:
        text = text[:MAX_CARACTERE]
    if text.count(".") < 5:
        return False
    return True


# ---------------------------------------------------------------------
# Constructia corpusului
# ---------------------------------------------------------------------

def construieste_corpus():
    corpus_final = []
    titluri_folosite = set()
    statistici = {"categorii_procesate": 0, "articole_luate": 0, "articole_respinse": 0}

    print("=== Extragere titluri din categorii ===")
    toate_titlurile = {}
    for categorie in CATEGORII:
        print(f"Categorie: {categorie}")
        titluri = obtine_titluri_din_categorie(categorie, ARTICOLE_PE_CATEGORIE)
        print(f"  -> {len(titluri)} titluri gasite")
        for t in titluri:
            if t not in titluri_folosite:
                toate_titlurile[t] = categorie
                titluri_folosite.add(t)
        statistici["categorii_procesate"] += 1

    print(f"\nTotal titluri unice colectate: {len(toate_titlurile)}")

    if not toate_titlurile:
        print("[X] Nu s-a colectat niciun titlu. Verifica conexiunea sau numele categoriilor.")
        return [], statistici

    print("=== Extragere continut articole ===")
    continuturi_existente = incarca_checkpoint()
    lista_titluri = list(toate_titlurile.keys())
    continuturi = obtine_continut_articole(lista_titluri, continuturi_existente)

    print("\n=== Curatare si filtrare calitate ===")
    for titlu, text_brut in continuturi.items():
        text_curat = curata_text(text_brut)
        if articol_e_de_calitate(text_curat):
            corpus_final.append(text_curat)
            statistici["articole_luate"] += 1
        else:
            statistici["articole_respinse"] += 1

    random.shuffle(corpus_final)
    return corpus_final, statistici


def salveaza_corpus(corpus, cale_fisier=FISIER_CORPUS):
    with open(cale_fisier, "w", encoding="utf-8") as f:
        for articol in corpus:
            f.write(articol)
            f.write("\n\n")

    total_cuvinte = sum(len(articol.split()) for articol in corpus)
    total_caractere = sum(len(articol) for articol in corpus)
    print(f"\nCorpus salvat in '{cale_fisier}'")
    print(f"  Articole incluse: {len(corpus)}")
    print(f"  Cuvinte totale (aprox.): {total_cuvinte:,}")
    print(f"  Caractere totale: {total_caractere:,}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    corpus, stats = construieste_corpus()
    print(f"\nStatistici finale: {stats}")

    if corpus:
        salveaza_corpus(corpus, FISIER_CORPUS)
    else:
        print("[X] Corpus gol — nu s-a salvat niciun fisier.")