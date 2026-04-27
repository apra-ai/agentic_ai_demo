# Agentic AI Research Assistant (LangGraph + Azure OpenAI)

## Überblick
Dieses Projekt implementiert eine prototypische Agentic AI-Anwendung auf Basis von Large Language Models. Ziel ist es, zentrale Konzepte agentischer Systeme – insbesondere Planung, Tool-Nutzung, Memory und iterative Entscheidungsprozesse – praktisch zu demonstrieren.

Der Agent arbeitet im ReAct-Stil und kann eigenständig entscheiden, ob externe Tools zur Beantwortung einer Anfrage genutzt werden.

---

## Architektur
Die Anwendung basiert auf folgenden Komponenten:

- **LLM (Azure OpenAI):** zentraler Reasoning-Kern  
- **LangGraph:** Orchestrierung des Agenten als Zustandsgraph  
- **LangChain:** Integration und Definition von Tools  
- **Tools:**
  - 🔍 Search Tool (z. B. Wikipedia)
  - 🧮 Calculator Tool
  - 📄 optional: Dokumenten-Retrieval
- **Memory/State:** Speicherung von Zwischenschritten und Kontext

Der Agent folgt einem iterativen Ablauf:
1. Analyse der Anfrage  
2. Entscheidung über Tool-Nutzung  
3. Ausführung von Aktionen  
4. Verarbeitung der Ergebnisse  
5. Generierung der finalen Antwort  

---

## Setup

### 1. Installation
```bash
pip install -r requirements.txt