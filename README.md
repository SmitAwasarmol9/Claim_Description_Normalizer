# 🧾 Claims Description Normalizer

An intelligent, AI-powered system designed to transform unstructured insurance claim descriptions into structured, actionable data while assessing potential fraud risk.

---

## 📌 Overview

Insurance claim processing often involves analyzing large volumes of unstructured text submitted by customers. This project addresses that challenge by leveraging modern AI techniques to:

- Standardize claim descriptions into structured JSON
- Extract key entities such as dates, locations, and assets
- Assess fraud risk using rule-based intelligence
- Generate professional reports for analysis

The system is built with scalability and real-world usability in mind, making it suitable for integration into insurance workflows, fraud detection systems, and automation pipelines.

---

## 🚀 Key Features

### 🔹 Intelligent Claim Structuring
- Converts raw text into structured JSON format
- Extracts:
  - Loss type
  - Severity level
  - Affected asset
  - Incident date & location
  - Summary & confidence score

### 🔹 Fraud Risk Analysis Engine
- Evaluates claims using multiple heuristics:
  - Missing or inconsistent information
  - Vague or suspicious language
  - Exaggeration indicators
  - Urgency patterns
- Outputs:
  - Fraud score (0–100)
  - Risk classification (Low / Medium / High)
  - Explainable flags

### 🔹 Named Entity Recognition (NER)
- Uses NLP to extract:
  - Dates
  - Locations
  - Organizations
  - Time references

### 🔹 PDF Support
- Upload and analyze claim documents directly
- Automatic text extraction from PDFs

### 🔹 Report Generation
- Generates downloadable, formatted PDF reports
- Includes structured data, fraud analysis, and entity insights

---

## 🛠 Technology Stack

| Component        | Technology Used |
|----------------|----------------|
| Frontend UI     | Streamlit |
| AI Processing   | Google Gemini API |
| NLP Engine      | spaCy |
| Backend Logic   | Python |
| PDF Handling    | PyPDF2, ReportLab |

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/SmitAwasarmol9/Claim_Description_Normalizer.git
cd Claim_Description_Normalizer

### Use Cases
Insurance claim automation
Fraud detection systems
Customer support analytics
Risk assessment pipelines
NLP-based document processing

🔒 Security Considerations
API keys are stored securely using environment variables
Sensitive data is excluded from version control via .gitignore
Designed with production-level practices in mind

👨‍💻 Author
Smit Awasarmol
Computer Science & Engineering Student
AI & Data Science Enthusiast
