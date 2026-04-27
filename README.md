# Agentic AI Research Assistant Demo

Eine minimale, aber saubere Python-Demo fuer einen Single-Agent Research Assistant mit Azure OpenAI, LangChain und LangGraph.

Die Demo zeigt die typischen Bausteine eines agentischen Systems:

- ein LLM als zentraler Controller
- Tool Use fuer Recherche, Berechnung und lokale Dokumente
- einen einfachen ReAct-aehnlichen Loop
- Memory und State im LangGraph-Zustand

## Projektstruktur

```text
agentic-research-demo/
├── main.py
├── graph.py
├── tools.py
├── config.py
├── requirements.txt
├── .env.example
├── README.md
└── docs/
    └── example.txt
```

## Architektur

### 1. LLM als Controller

In `graph.py` wird ein `AzureChatOpenAI`-Modell initialisiert und mit Tools verbunden. Das Modell entscheidet in jedem Schritt, ob es direkt antworten kann oder ein Tool aufrufen sollte.

### 2. LangGraph als Zustandsmaschine

Der Agent wird als einfacher `StateGraph` modelliert:

- Node `agent`: ruft das LLM auf
- Node `tools`: fuehrt Tool Calls aus
- Conditional Edge: entscheidet, ob weitere Tool-Nutzung notwendig ist oder der Lauf beendet wird

Gespeichert werden unter anderem:

- `question`
- `messages`
- `intermediate_steps`
- `tool_outputs`
- `used_tools`
- `final_answer`
- `iterations`

### 3. Tools

- `search_tool`: Wikipedia-Suche fuer Fakten und Hintergrundinformationen
- `calculator_tool`: sichere Auswertung mathematischer Ausdruecke mit eingeschraenktem AST
- `document_retrieval_tool`: einfache Suche in lokalen `.txt`-Dateien im Ordner `docs/`

## Ablauf

1. Der Nutzer stellt eine Frage.
2. Der `agent`-Node bewertet die Anfrage.
3. Falls noetig, erzeugt das Modell einen Tool Call.
4. Der `tools`-Node fuehrt das Tool aus und schreibt die Observation in den State.
5. Der Agent bewertet erneut, ob weitere Schritte noetig sind.
6. Sobald genug Informationen vorliegen oder das Iterationslimit erreicht ist, wird eine finale Antwort erzeugt.

## Stop Condition

Die Demo beendet den Loop, wenn eine dieser Bedingungen erreicht ist:

- das Modell liefert eine finale Antwort ohne weiteren Tool Call
- das maximale Iterationslimit ist erreicht, standardmaessig `5`

## Setup

### Voraussetzungen

- Python 3.11 oder neuer
- ein Azure OpenAI Deployment fuer ein Chat-Modell mit Tool Calling

### 1. Abhaengigkeiten installieren

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Umgebungsvariablen konfigurieren

Kopiere `.env.example` nach `.env` und trage deine Werte ein:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
```

### 3. Demo starten

Interaktiv:

```bash
python main.py
```

Direkt mit Frage:

```bash
python main.py "Welche Firma hatte 2023 mehr Umsatz: Tesla oder BMW? Berechne den Unterschied."
```

## Beispielanfragen

```bash
python main.py "How many years are there between the founding of Google and Facebook?"
python main.py "What is the population difference between Paris and Rome?"
```

## Erwartete CLI-Ausgabe

Die Anwendung gibt drei Bereiche aus:

- finale Antwort
- verwendete Tools
- Zwischenschritte des Agenten

Beispielhaft:

```text
=== Final Answer ===
BMW hatte 2023 den hoeheren Umsatz. Der Unterschied betraegt ...

=== Used Tools ===
search_tool, calculator_tool

=== Intermediate Steps ===
1. Agent: decided to call tool(s): search_tool
2. Tool search_tool: input={'query': 'Tesla revenue 2023'} | output=...
3. Agent: decided to call tool(s): search_tool, calculator_tool
...
```

## Dateien im Detail

- `config.py`: laedt und validiert Azure-OpenAI-Konfiguration aus `.env`
- `tools.py`: enthaelt alle Tools fuer Suche, Rechnen und Dokumentzugriff
- `graph.py`: definiert den LangGraph-Workflow und den Agent-State
- `main.py`: einfache CLI zum Starten der Demo

## Hinweise zur Seminar-Demo

Diese Implementierung ist bewusst einfach gehalten:

- keine Multi-Agent-Architektur
- kein komplexes Planning-Modul
- keine Vektordatenbank
- keine Persistenz ueber einen Prozesslauf hinaus

Dadurch bleibt der Fokus auf dem eigentlichen agentischen Kern:

- Reasoning Loop
- Tool Selection
- Observation Handling
- State Management mit LangGraph

## Moegliche Erweiterungen

- persistenter Checkpointer fuer LangGraph
- bessere lokale Dokumentensuche mit Embeddings
- strukturierte Ausgabe als JSON
- Streaming fuer die CLI