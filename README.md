# 🤖 Agentic AI System for Enterprise Cost Optimization

> An autonomous AI system that continuously monitors ecommerce operations, detects cost leakages, identifies root causes, takes corrective actions, validates outcomes, and improves itself over time. The system behaves like a **self-driving business analyst** that actively fixes problems and proves its impact.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Flask](https://img.shields.io/badge/Flask-2.3.0+-lightgrey.svg)](https://flask.palletsprojects.com/)

---

## 🚀 Features

- **Autonomous Operation** — No manual dashboard analysis required
- **Dynamic Schema Understanding** — Adapts to any dataset structure using LLM
- **Multi-Level Anomaly Detection** — Pinpoints issues at product, seller, category, and time levels
- **Intelligent Action Generation** — Suggests concrete, executable fixes with expected impact
- **Validation & Feedback Loop** — Learns from past actions to improve future decisions
- **Web Dashboard** — User-friendly interface to monitor results and interact with the system

---

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER UPLOADS CSV FILES                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SCHEMA UNIFICATION AGENT                      │
│            (LLM-powered: detects columns, merges datasets)      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BUSINESS ANALYST AGENT                        │
│         (Calculates KPIs, trends, segments, cost leakages)      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ANOMALY DETECTION AGENT                       │
│          (Identifies specific problematic products/sellers)     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MANAGER AGENT                             │
│         (Orchestrates action & validation, maintains memory)    │
└─────────────────────────────────────────────────────────────────┘
                        │               │
                        ▼               ▼
            ┌─────────────────┐ ┌─────────────────┐
            │  ACTION AGENT   │ │VALIDATION AGENT │
            │ (Suggests fixes)│ │(Checks results) │
            └─────────────────┘ └─────────────────┘
                        │               │
                        └───────┬───────┘
                                ▼
                    ┌─────────────────────┐
                    │    FEEDBACK LOOP    │
                    │  (Self-improvement) │
                    └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core language |
| **LangChain & LangGraph** | Agent orchestration |
| **Groq LLM** (Mixtral 8x7B) | Fast LLM inference |
| **Pandas & NumPy** | Data processing |
| **Flask** | Web dashboard backend |
| **Chart.js** | Data visualization |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Vinil30/ecommerce-agentic-ai.git
cd ecommerce-agentic-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
groq
langgraph
langchain_groq
langchain
pymongo
python-dotenv
pandas
flask
werkzeug
```

### 4. Set Up API Key

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
MONGO_URI="..."
FLASK_SECRET_KEY
```

> Get your free API key from [Groq Console](https://console.groq.com)

### 5. Download Sample Dataset (Optional)

The system works with any ecommerce dataset. For testing, use the [Olist Brazilian Ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) dataset:

```bash
kaggle datasets download olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip -d Datasets/
```

---

## 🚀 Usage

### Web Dashboard

```bash
python app.py
```

Navigate to **http://localhost:5000**

### Command Line

```python
from graph import create_app
from utils.schema_unifier import SchemaUnificationAgent
from utils.business_analyst import BusinessAnalystAgent
from utils.anomaly_detector import AnomalyDetectionAgent
from utils.manager_agent import ManagerAgent

# Initialize with API key
api_key = "your-groq-api-key"

# Unify datasets
unifier = SchemaUnificationAgent(api_key=api_key)
unifier.unify(["orders.csv", "items.csv", "payments.csv"])

# Analyze
ba = BusinessAnalystAgent(api_key=api_key, unified_dataset_path="unified_dataset.csv")
ba_result = ba.analyze()

# Detect anomalies
anomaly = AnomalyDetectionAgent(api_key=api_key, enriched_dataset_path="enriched_dataset.csv")
anomaly_result = anomaly.analyze()

# Manager decides next steps
manager = ManagerAgent(api_key=api_key)
decision = manager.decide(ba_result, anomaly_result)
```

---

## 📊 Agent Descriptions

### 1. Schema Unification Agent
- Reads multiple CSV files
- Uses LLM to identify column meanings
- Merges datasets intelligently
- Outputs a unified dataset

### 2. Business Analyst Agent
- Calculates KPIs (revenue, profit, margins)
- Generates time series trends
- Performs segment analysis
- Identifies aggregate cost leakages

### 3. Anomaly Detection Agent
- Pinpoints problematic products (high return rates, low margins)
- Flags problematic sellers (low reviews, defective products)
- Identifies problematic categories (systemic issues)
- Detects time-based anomalies (sudden drops/spikes)

### 4. Action Agent
- Generates concrete fixes (pause seller, adjust pricing, negotiate freight)
- Provides expected financial impact
- Returns structured, executable actions

### 5. Validation Agent
- Compares before/after metrics
- Determines if action was effective
- Suggests adjustments if needed

### 6. Manager Agent
- Orchestrates all agents
- Maintains action history
- Uses LLM to decide when to validate vs act
- Self-improves over time

---

## 📁 Project Structure

```
EnterpriseAutomation/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
├── Datasets/                   # Sample datasets
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── DataUnifier.py       # Schema Unification Agent
│   ├── DatasetAnalyser.py     # Business Analyst Agent
│   ├── AnamolyDetection.py     # Anomaly Detection Agent
│   ├── ActionAgent.py         # Action Agent
│   ├── ValidationAgent.py     # Validation Agent
│   ├── ManagerAgent.py        # Manager Agent
│   └── Graph.py       # LangGraph workflow definition with Helper functions
├── templates/
│   ├── index.html              # Dashboard home
│   ├── dashboard.html             # File upload page and analysis results
│   ├── signin.html            # Signin Page
│   ├── signup.html            # Sign Up page
│   └── history.html            # Action and Analysis history

```

---

## 🔄 Workflow

1. **Upload** — User uploads multiple CSV files
2. **Unify** — Schema Unification Agent merges into single dataset
3. **Analyze** — Business Analyst calculates KPIs and trends
4. **Detect** — Anomaly Detection finds specific issues
5. **Decide** — Manager Agent determines next steps
6. **Act** — Action Agent generates fixes if needed
7. **Validate** — After new data, validation checks effectiveness
8. **Learn** — System stores outcomes for future improvement

---

## 📈 Sample Output

```
📊 BUSINESS ANALYSIS RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Total Revenue:    $20,579,664.01
📈 Profit Margin:    88.5%
📦 Total Orders:     99,441
💸 Cost Leakage:     $5,348,892.87

🔴 TOP PROBLEMATIC PRODUCTS
   Product: ee3d532c8a4...  | Loss: $36,393 | Return Rate: 78%
   Product: 25c38557cf7...  | Loss: $24,513 | Return Rate: 60%

🔵 TOP PROBLEMATIC SELLERS
   Seller: 7c67e1448b0...   | Loss: $102,529 | 414 low-review products

💡 INSIGHTS
   - Current performance: Revenue $20.58M with 88.5% margin
   - Biggest problem: $5.35M cost leakage from high return rates
   - Recommendation: Pause problematic sellers with >400 low-review products
````
---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 Commit History

| Date | Commit Message |
|---|---|
| Mar 29, 2026 | Finalising Dashboard and History details |
| Mar 29, 2026 | Updated Dashboard and History fetching |
| Mar 28, 2026 | Adding Flask routes |
| Mar 28, 2026 | Updating all python files after testing |
| Mar 27, 2026 | Adding datasets for testing, integrated all agents into LangGraph file |
| Mar 27, 2026 | Adding templates folder with html files |
| Mar 27, 2026 | Adding Unifier agent for handling multiple csv files |
| Mar 27, 2026 | Added Action Validator Agent |
| Mar 27, 2026 | Added Business Analyzer Agent |
| Mar 27, 2026 | Added Manager Agent |
| Mar 27, 2026 | Added Anomaly Detector |
| Mar 27, 2026 | Added Action Agent |
| Mar 26, 2026 | Updating Langgraph flow |
| Mar 25, 2026 | Adding Langgraph |

---

## Acknowledgments

- [Groq](https://groq.com) — for fast LLM inference
- [LangChain](https://langchain.com) — for the agent framework
- [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — for the sample dataset

---

## Contact

**Vinil30** — [GitHub Profile](https://github.com/Vinil30)
**Project Link:** [https://github.com/Vinil30/EnterpriseAutomation](https://github.com/Vinil30/EnterpriseAutomation)
**Email** - (rapellysaivinil@gmail.com)
