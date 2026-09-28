"""
Preprocesare agresiva a corpusului text pentru antrenare LSTM char-level.

Reduce textul la vocabular minim: litere minuscule (inclusiv diacritice
romanesti), spatiu, si un token special "eos" care marcheaza sfarsitul
de propozitie (inlocuieste . ! ?). Restul punctuatiei, cifrele,
parantezele si orice alt simbol sunt eliminate complet.

Vocabular rezultat asteptat: ~32 caractere (26 litere + ă â î ș ț + spatiu)
Tokenul "eos" e format din litere deja existente in vocabular (e, o, s),
deci nu adauga niciun caracter nou - doar o secventa recognoscibila
pe care modelul o poate invata ca marker de granita.
"""

import re
import unicodedata
from collections import Counter

FISIER_INTRARE = "corpus_ro.txt"
FISIER_IESIRE = "corpus_ro_clean.txt"
FISIER_RAPORT = "raport_vocabular.txt"

LUNGIME_MIN_PARAGRAF = 50


# ---------------------------------------------------------------------
# 1. Normalizare Unicode (NFC)
# ---------------------------------------------------------------------

def normalizeaza_unicode(text):
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------
# 2. Normalizare diacritice
# ---------------------------------------------------------------------

HARTA_DIACRITICE = {
    "\u015f": "\u0219",  # ş -> ș
    "\u015e": "\u0219",  # Ş -> ș
    "\u0163": "\u021b",  # ţ -> ț
    "\u0162": "\u021b",  # Ţ -> ț
}


def normalizeaza_diacritice(text):
    for gresit, corect in HARTA_DIACRITICE.items():
        text = text.replace(gresit, corect)
    return text


# ---------------------------------------------------------------------
# 3. Inserare token EOS la sfarsit de propozitie
# ---------------------------------------------------------------------
# Se face INAINTE de stergerea punctuatiei, pentru ca altfel informatia
# despre unde se termina o propozitie s-ar pierde definitiv.
#
# Detectam . ! ? (eventual repetate, ex. "?!" sau "...") si le inlocuim
# cu " EOS " - cu spatii in jur, ca sa ramana un cuvant separat, nu lipit
# de cuvintele vecine. Dupa lowercase, devine " eos ".
#
# Limitare de mentionat: regex-ul e simplu, deci trateaza si punctele din
# abrevieri (ex. "dr.") ca sfarsit de propozitie. Pentru text din Wikipedia,
# asta e un compromis acceptabil - abrevierile sunt relativ rare fata de
# volumul total de propozitii.

PATTERN_SFARSIT_PROPOZITIE = re.compile(r'[.!?]+["\')\]]*')


def insereaza_eos(text):
    return PATTERN_SFARSIT_PROPOZITIE.sub(" EOS ", text)


# ---------------------------------------------------------------------
# 4. Lowercase
# ---------------------------------------------------------------------

def converteste_minuscule(text):
    return text.lower()


# ---------------------------------------------------------------------
# 5. Deduplicare la nivel de paragraf
# ---------------------------------------------------------------------

def elimina_duplicate(text_brut_cu_paragrafe):
    paragrafe = text_brut_cu_paragrafe.split("\n\n")
    vazute = set()
    unice = []
    for p in paragrafe:
        cheie = p.strip().lower()
        if cheie and cheie not in vazute:
            vazute.add(cheie)
            unice.append(p)
    return "\n\n".join(unice)


# ---------------------------------------------------------------------
# 6. Eliminare TOT ce nu e litera romaneasca sau spatiu
# ---------------------------------------------------------------------
# Tokenul "eos" supravietuieste automat acestui filtru pentru ca e format
# din litere e, o, s - deja in whitelist. Nu necesita tratament special.

LITERE_VALIDE = set("abcdefghijklmnopqrstuvwxyzăâîșț ")


def pastreaza_doar_litere_si_spatiu(text):
    text = text.replace("\n", " ").replace("\t", " ")
    rezultat = []
    for ch in text:
        if ch in LITERE_VALIDE:
            rezultat.append(ch)
        else:
            rezultat.append(" ")
    return "".join(rezultat)


# ---------------------------------------------------------------------
# 7. Normalizare spatii
# ---------------------------------------------------------------------

def normalizeaza_spatii(text):
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------
# 8. Filtrare fragmente prea scurte
# ---------------------------------------------------------------------

def filtreaza_fragmente_scurte(text_brut_cu_paragrafe):
    paragrafe = text_brut_cu_paragrafe.split("\n\n")
    pastrate = []
    for p in paragrafe:
        p_curatat = normalizeaza_spatii(pastreaza_doar_litere_si_spatiu(p))
        if len(p_curatat) >= LUNGIME_MIN_PARAGRAF:
            pastrate.append(p_curatat)
    return " ".join(pastrate)


# ---------------------------------------------------------------------
# 9. Raport vocabular
# ---------------------------------------------------------------------

def genereaza_raport_vocabular(text, cale_fisier=FISIER_RAPORT):
    contor = Counter(text)
    total_caractere = len(text)
    vocab_size = len(contor)
    numar_eos = text.count(" eos ") + (1 if text.startswith("eos ") else 0)

    with open(cale_fisier, "w", encoding="utf-8") as f:
        f.write(f"Total caractere: {total_caractere:,}\n")
        f.write(f"Vocabular unic: {vocab_size} caractere\n")
        f.write(f"Aparitii token EOS: ~{numar_eos:,}\n\n")
        f.write("Frecventa caracterelor (descrescator):\n")
        f.write("-" * 40 + "\n")
        for ch, frecventa in contor.most_common():
            afisare = "SPATIU" if ch == " " else ch
            procent = 100 * frecventa / total_caractere
            f.write(f"{afisare:10} {frecventa:>10,}  ({procent:5.2f}%)\n")

    print(f"[i] Raport vocabular salvat in '{cale_fisier}'")
    print(f"[i] Vocabular unic: {vocab_size} caractere distincte")
    print(f"[i] Aparitii token EOS: ~{numar_eos:,}")

    if vocab_size > 32:
        neasteptate = [ch for ch in contor if ch not in LITERE_VALIDE]
        print(f"[!] Vocabular peste 32 - caractere neasteptate ramase: {neasteptate}")

    return contor


# ---------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------

def preproceseaza(text):
    print("[1/8] Normalizare Unicode (NFC)...")
    text = normalizeaza_unicode(text)

    print("[2/8] Normalizare diacritice...")
    text = normalizeaza_diacritice(text)

    print("[3/8] Inserare token EOS la sfarsit de propozitie...")
    text = insereaza_eos(text)

    print("[4/8] Conversie la minuscule...")
    text = converteste_minuscule(text)

    print("[5/8] Deduplicare paragrafe...")
    text = elimina_duplicate(text)

    print("[6/8] Filtrare fragmente scurte + eliminare punctuatie/cifre/simboluri...")
    text = filtreaza_fragmente_scurte(text)

    print("[7/8] Normalizare finala a spatiilor...")
    text = normalizeaza_spatii(text)

    print("[8/8] Gata.")
    return text


def main():
    print(f"Citesc '{FISIER_INTRARE}'...")
    with open(FISIER_INTRARE, "r", encoding="utf-8") as f:
        text_brut = f.read()

    caractere_initiale = len(text_brut)
    cuvinte_initiale = len(text_brut.split())
    print(f"Text brut: {caractere_initiale:,} caractere, ~{cuvinte_initiale:,} cuvinte\n")

    text_curat = preproceseaza(text_brut)

    caractere_finale = len(text_curat)
    cuvinte_finale = len(text_curat.split())

    with open(FISIER_IESIRE, "w", encoding="utf-8") as f:
        f.write(text_curat)

    print(f"\nCorpus normalizat salvat in '{FISIER_IESIRE}'")
    print(f"  Caractere: {caractere_initiale:,} -> {caractere_finale:,} "
          f"({100 * caractere_finale / caractere_initiale:.1f}% pastrat)")
    print(f"  Cuvinte:   {cuvinte_initiale:,} -> {cuvinte_finale:,} "
          f"({100 * cuvinte_finale / cuvinte_initiale:.1f}% pastrat)")

    genereaza_raport_vocabular(text_curat)


if __name__ == "__main__":
    main()