# VeriFact AI Session Notes and Prompt Pack

Date: 2026-04-14

This file captures the current working direction of the project from the session thread:
- Research -> architecture -> product pipeline
- Hybrid hallucination detection and correction stack
- Multi-agent prompting strategy for Claude
- Practical constraints for free/open deployment

## 1. Reality Check

Research papers provide:
- Theory on hallucination causes
- Detection methods (uncertainty, NLI, classifiers, benchmarks)
- Mitigation patterns (RAG, verification, correction)

Research papers do not provide:
- Complete API/service architecture
- Browser extension integration details
- Real-time latency handling and caching strategy
- Product-grade deployment and operations runbooks

Conclusion:
You must convert research -> system design -> implementation -> validation.

## 2. Product Definition

Target product:
A real-time AI Truth Verification Layer that:
- Detects hallucination risk in chatbot outputs
- Highlights red flags and potential bias
- Suggests corrections with evidence/source links
- Produces confidence scores
- Works through browser extension + local/hosted backend

## 3. Hybrid Architecture Blueprint

### Step 1: Input and Claim Extraction
- Input: LLM output text (ChatGPT, Claude, Gemini, etc.)
- Output: atomic factual claims

### Step 2: Retrieval Layer (RAG)
- Retrieve evidence from trusted corpora
- Stack: FAISS + sentence embeddings

### Step 3: Verification Core
Combine:
- Semantic matching (claim-evidence relevance)
- Uncertainty/self-consistency signal (multi-sample variance)
- NLI contradiction/entailment scoring

### Step 4: Confidence Fusion
- Weighted score from retrieval, NLI, uncertainty, source trust
- Output normalized to 0-100

### Step 5: Correction and Attribution
- For contradicted/high-risk claims, generate correction candidates
- Attach evidence snippets and source links

### Step 6: UX Output Layer
- Green: supported
- Amber: weak evidence/unverifiable
- Red: contradiction/hallucination risk
- Expose reason, confidence, evidence, and source

## 4. Free/Open Stack Strategy

Backend:
- Python API (current local overlay server)
- Optional FastAPI migration for hosted deployment

Models:
- Local models via Ollama and open-source HF models
- NLI model for contradiction detection

Retrieval:
- FAISS local index + trusted source corpus

Frontend:
- Current Streamlit dashboard for operator view
- Browser extension (Manifest V3) for in-chat analysis UI

Hosting:
- GitHub for source + CI + releases
- Optional free runtime: Render/Fly/Railway style service for API
- Vercel best suited for frontend/static layers, not heavy Python inference backend

## 5. Known Hard Problems

- Real-time latency under model cold starts
- Ambiguous/non-binary truth claims
- Source reliability scoring quality
- Multi-hop fact verification
- Bias detection false positives

## 6. High-ROI Build Sequence

1. MVP backbone:
- claim extraction
- retrieval
- NLI verification
- confidence fusion

2. Intelligence upgrades:
- self-consistency/uncertainty scoring
- source trust ranking
- rationale extraction

3. Product experience:
- extension live alerts
- actionable correction cards
- desktop app style shell

4. Evaluation and publication quality:
- benchmark harness
- ablations by module
- calibration and threshold reports

## 7. Enhanced Claude Multi-Agent Prompt (Worker + Supervisor + RALPH)

Use this prompt when handing architecture synthesis to Claude.

---

You are orchestrating a research-engineering multi-agent team to design and implement an AI Truth Layer system.

OBJECTIVE:
Build a robust, deployable, research-grounded system that detects hallucinations, verifies claims, suggests corrections with citations, and outputs calibrated confidence.

MANDATORY TEAM STRUCTURE:
For each task, assign two agents:
1) Worker Agent: produces first-pass output
2) Supervisor Agent: audits, critiques, and improves output

GLOBAL TEAMS:
1. Research Extraction Team
2. Cross-Paper Synthesis Team
3. Architecture Team
4. Algorithm Team
5. Implementation Team
6. Risk/Failure Team
7. Optimization Team

RALPH LOOP (must run per module):
- Reason: draft solution
- Analyze: identify flaws and assumptions
- Learn: extract improvement actions
- Plan: revise architecture/algorithm
- Harmonize: produce validated final output
Repeat until major flaws are resolved.

STRICT RULES:
- No fabricated claims, metrics, URLs, or citations
- Separate verified facts from assumptions
- Label any engineering decision not directly paper-backed
- Prioritize correctness and reproducibility over stylistic output

REQUIRED OUTPUTS:
1. Paper-to-module mapping table
2. Final hybrid architecture diagram description (textual)
3. Module-level pseudocode and data contracts
4. API endpoints and request/response schemas
5. Runtime/deployment architecture (local + optional cloud)
6. Failure modes and mitigations
7. Latency/cost optimization plan
8. MVP -> production roadmap
9. Verification checklist with pass/fail criteria

SYSTEM MODULES TO DELIVER:
- Input normalization
- Claim extraction
- Retrieval/ranking
- Verification (NLI + uncertainty + semantic checks)
- Confidence calibration/fusion
- Correction and citation generation
- Output formatter (dashboard + extension payload)

FREE STACK CONSTRAINT:
Use open-source models, free tools, and no paid dependencies by default.

---

## 8. Current Session Signals

- "Analyzer offline: TypeError: Failed to fetch" in extension generally indicates local API not reachable from browser context at that moment (service down, cold start, or blocked request).
- Current implementation has been upgraded to include native desktop shell launch, extension alert chips, and structured alert payloads.
- For browser-side changes, always reload extension after content/script updates.

## 9. Immediate Next-Day Execution Plan

1. Run live matrix tests on ChatGPT, Claude, Gemini
2. Capture latency and alert-quality metrics per platform
3. Improve classifier-backed bias detection (replace simple lexical heuristic)
4. Add uncertainty/self-consistency scoring into confidence fusion
5. Finalize GitHub publishing and deployment docs
