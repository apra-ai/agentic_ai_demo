# Agentic AI Research Assistant Demo

Eine minimale, aber saubere Python-Demo fuer einen Single-Agent Research Assistant mit Azure OpenAI, LangChain und LangGraph.

Die Demo zeigt die typischen Bausteine eines agentischen Systems:

- ein LLM als zentraler Controller
- Tool Use fuer Recherche, Berechnung und lokale Dokumente
- einen einfachen ReAct-aehnlichen Loop
- Memory und State im LangGraph-Zustand
- explizite Sicht auf Planung, Memory, Reasoning und Entscheidungslogik

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
- `plan`
- `intermediate_steps`
- `tool_outputs`
- `used_tools`
- `reasoning_log`
- `decision_log`
- `memory_log`
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

## Bezug zur Seminararbeit

Die Demo bildet die Komponenten aus Kapitel 3.3 direkt in der CLI und im LangGraph-State ab:

- `Planung`: Zu Beginn wird aus der Nutzerfrage ein einfacher Arbeitsplan erzeugt und unter `=== Plan ===` ausgegeben.
- `Tool-Nutzung`: Tool Calls und Tool Outputs bleiben sichtbar und werden unter `=== Used Tools ===` und `=== Intermediate Steps ===` protokolliert.
- `Memory`: Der State speichert Frage, Observations, verwendete Tools und Memory-Snapshots. Diese Kurzzeit-Memory erscheint unter `=== Memory ===`.
- `Reasoning`: Die Demo protokolliert, warum der Agent weitere Informationen benoetigt oder warum er zum finalen Antwortschritt uebergeht. Diese Eintraege erscheinen unter `=== Reasoning ===`.
- `Entscheidungslogik`: Konkrete Entscheidungen wie `call search_tool`, `call calculator_tool` oder `stop and answer` werden unter `=== Decision Log ===` sichtbar gemacht.

Damit zeigt die Demo nicht nur das Endergebnis, sondern auch die internen Komponenten, auf die sich deine Seminararbeit bezieht.

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
python main.py "How many years are there between the founding of Google and Facebook?"
```

## Beispielanfragen

```bash
python main.py "How many years are there between the founding of Google and Facebook?"
python main.py "What is the population difference between Paris and Rome?"
```

## Erwartete CLI-Ausgabe

Die Anwendung gibt mehrere Bereiche aus:

- Plan
- finale Antwort
- verwendete Tools
- Memory
- Reasoning
- Entscheidungslogik
- Zwischenschritte des Agenten

Beispielhaft:

```text
=== Plan ===
1. Analyse the question and identify which facts or numbers are needed.
2. Collect evidence for the relevant entities before answering.
3. Use the calculator tool to compute the requested value from the collected numbers.
4. Synthesize the observations into a concise final answer with a short uncertainty note if needed.

=== Final Answer ===
BMW hatte 2023 den hoeheren Umsatz. Der Unterschied betraegt ...

=== Used Tools ===
search_tool, calculator_tool

=== Memory ===
1. Iteration 0 | question='Welche Firma hatte 2023 mehr Umsatz: Tesla oder BMW?...' | observations=0 | used_tools=[] | next='agent analysis'

=== Reasoning ===
1. Reasoning: the current state does not yet support a reliable final answer, so the agent requests search_tool.

=== Decision Log ===
1. Decision: call search_tool next.

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
- kein komplexes Planning-Modul, sondern ein bewusst einfaches Plan-Schema fuer die Demo
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