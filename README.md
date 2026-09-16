# 🦙 Enterprise RAG systém s důrazem na soukromí (Ollama & LangChain)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2-green.svg)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange.svg)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-red.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source RAG agent navržený pro **100% lokální a bezpečnou analýzu PDF dokumentů**. Žádná data neopouštějí vaši lokální infrastrukturu. Postaveno na technologiích **LangChain**, **Ollama**, **ChromaDB** a **Streamlit**.

---

## 🌟 Hlavní funkce

- **🔒 100% On-Premise & Offline:** Běží kompletně na lokálních modelech v Ollamě (`llama3.2` + `nomic-embed-text`).
- **📚 Ověřitelné citace zdrojů:** Každá vygenerovaná odpověď obsahuje přesné názvy souborů a čísla stránek.
- **⚡ Podpora dvou providerů:** Snadné přepínání mezi lokální Ollamou a cloudovým Azure OpenAI.
- **🎨 Interaktivní UI:** Moderní Streamlit rozhraní s real-time streamováním odpovědí a historií chatu.
- **📦 Perzistentní vektorové úložiště:** Využívá ChromaDB pro rychlé vektorové vyhledávání.

---

## 🏗️ Architektura systému

```text
[ Nahrání PDF ] ──► [ PyPDF Loader ] ──► [ Text Splitter ] ──► [ nomic-embed-text ]
                                                                    │
                                                                    ▼
[ Dotaz uživatele ] ──► [ Top-K Vyhledání ] ◄───────────── [ ChromaDB Vektorová DB ]
                              │
                              ▼
                   [ Llama 3.2 / Ollama ] ──► [ Streamlit UI s citacemi ]
```

---

## 🚀 Rychlý návod k spuštění

### 1. Stažení potřebných lokálních modelů

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Příprava prostředí a spuštění

```bash
git clone https://github.com/marian19commits/enterprise-local-rag.git
cd enterprise-local-rag
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Struktura projektu

```text
enterprise-local-rag/
├── src/
│   ├── __init__.py
│   ├── config.py         # Konfigurace aplikace a prostředí
│   └── rag_engine.py     # RAG logika a indexace do ChromaDB
├── app.py                # Streamlit uživatelské rozhraní
├── requirements.txt      # Závislosti projektu
├── .env.example          # Šablona pro proměnné prostředí
├── .gitignore            # Ignorované soubory
└── README.md             # Dokumentace projektu
```

---

## 📜 Licence

Tento projekt je licencován pod licencí MIT.
