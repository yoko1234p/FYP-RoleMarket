# Project Brief: AI-Driven Market-Informed Character IP Design Extension and Demand Forecasting System

**Document Version:** 1.0
**Date:** 2025-01-25
**Author:** Product Manager
**Status:** Draft for Review

---

## Executive Summary

This Final Year Project (FYP) develops a **commercial-grade AI system** that real character IP design companies can deploy immediately for market-informed design and demand forecasting. The system integrates Google Trends analysis, LLM-based prompt generation, **Midjourney API via TTAPI**, CLIP feature extraction, and LSTM demand forecasting into a production-ready pipeline accessible through a Streamlit web application.

**Product Concept:** A business-ready tool that automates the entire workflow from trend monitoring to sales forecasting, matching how industry leaders like ToyzeroPlus actually use AI tools in production (they use Midjourney for design exploration).

**Primary Problem:** Character IP design companies face challenges in rapidly creating seasonal character variations that align with current market trends while accurately forecasting demand to minimize financial risk and optimize inventory.

**Target Market:** Character IP design companies (ToyzeroPlus validation case using open-source Pikachu for demonstration), with immediate commercial viability for the broader toy design industry.

**Key Value Proposition:** Production-ready system using industry-standard tools (Midjourney API) that can be deployed immediately after FYP, reducing design iteration time from weeks to hours with data-driven demand forecasts.

---

## Problem Statement

### Current State and Pain Points

ToyzeroPlus and similar character IP companies face three critical operational challenges:

1. **Slow Trend Response:** Manual market research and design iteration takes 3-4 weeks, causing companies to miss trending themes by the time products reach market.

2. **Inconsistent Character Extensions:** Creating seasonal variations while maintaining brand consistency requires experienced designers and multiple revision cycles, creating bottlenecks in the creative process.

3. **Demand Forecasting Uncertainty:** Production decisions rely heavily on intuition and limited historical data, leading to either overproduction (financial loss) or underproduction (missed revenue opportunities).

### Impact of the Problem

- **Financial Risk:** Inventory mismatch costs can reach 20-30% of production budget
- **Missed Opportunities:** Delayed response to trends results in lost market share during peak seasonal periods
- **Resource Inefficiency:** Design teams spend 60-70% of time on mechanical variations rather than innovative concepts
- **Decision Anxiety:** Leadership lacks quantitative data to support production volume decisions

### Why Existing Solutions Fall Short

- **Traditional Design Tools:** Photoshop/Illustrator require manual execution for every variation, offering no trend intelligence or consistency automation
- **Standalone AI Tools:** Companies like ToyzeroPlus use Midjourney for design exploration but manually handle prompt engineering, trend integration, and demand forecasting separately—no unified workflow
- **Demand Forecasting Software:** Enterprise solutions (SAP, Oracle) require extensive historical sales data and don't integrate design features
- **Trend Analysis Tools:** Google Trends and social media analytics provide raw data but don't translate insights into actionable design prompts

**Industry Validation:** ToyzeroPlus confirmed they actively use Midjourney for character design iteration, proving commercial demand for AI-assisted design tools. However, they lack integration with trend monitoring and demand forecasting—exactly what this system provides.

### Urgency and Importance

The toy industry's shift toward rapid seasonal releases (4-6 collections per year vs. traditional 2) demands faster, more data-informed design-to-production pipelines. ToyzeroPlus must adopt AI-driven workflows to remain competitive in the evolving market landscape, making this solution critical for near-term survival and growth.

---

## Proposed Solution

### Core Concept and Approach

We propose a four-stage AI pipeline that transforms market trends into production-ready design forecasts:

**Stage 1 - Trend Intelligence (Objective 1):**
- Monitor Google Trends for seasonal keywords (e.g., "春節", "可愛", "紅色")
- Extract top trending keywords using TF-IDF
- Generate detailed SDXL prompts via GPT_API_free that merge character base descriptions with trend elements

**Stage 2 - Commercial-Grade Design Generation (Objective 2):**
- Integrate TTAPI Midjourney API for production-quality image generation
- Generate 4 design variations per seasonal theme using Midjourney's character reference (--cref) feature
- Select 1-2 high-quality Pikachu reference images for character consistency across all generations
- Validate consistency using CLIP similarity scores (>0.75 core features, >0.60 style)
- Total cost: ~$10-30 for TTAPI quota (28-40 image generations)

**Stage 3 - Demand Prediction (Objective 3):**
- Build hybrid LSTM model combining time-series trends (past 3-4 seasons) with static design features (CLIP embeddings)
- Train on 60 simulated historical data points (5 years × 4 seasons × 3 designs) using rule-based simulation incorporating Google Trends, CLIP similarity, production constraints, and seasonal factors
- Predict sales volume for new design variations

**Stage 4 - Integrated Web Application (Objective 4):**
- Deploy Streamlit application with two core pages:
  - Design Generation: Input trends → Display 4 mood board variations
  - Forecast Dashboard: Select season → View predicted sales + trend insights
- Integrate all components via TTAPI Midjourney API and local LSTM inference

### Key Differentiators

1. **End-to-End Integration:** Unlike fragmented tools, our system closes the loop from trend analysis to demand forecast
2. **Industry-Standard Tools:** Uses Midjourney API (proven commercial tool that ToyzeroPlus already uses), not experimental academic models
3. **Hybrid Forecasting Architecture:** Combines market trends (temporal) with design features (static) for more accurate predictions
4. **Production-Ready Design:** Built with real business workflows in mind—can be deployed commercially after FYP
5. **Commercial Viability:** Modest API costs ($10-30) make this economically feasible for real toy companies

### Why This Solution Will Succeed

- **Industry-Proven Tools:** Midjourney API already used by ToyzeroPlus in production—we're automating their existing workflow, not introducing unproven technology
- **Realistic Scope:** 17-18 day timeline focuses on MVP delivery, and Midjourney API integration is faster than training custom models (saves 2-3 days vs LoRA approach)
- **Commercial Validation:** ToyzeroPlus confirmed they use Midjourney for design, proving market demand for this exact capability
- **Hybrid Human-AI Workflow:** System generates mood boards for human decision-making, not full automation (reducing adoption resistance)
- **Cost-Effective:** $10-30 total cost for FYP demonstration proves economic viability for real businesses

### High-Level Vision

Transform character IP development from intuition-driven to intelligence-augmented, where designers focus on creative direction while AI handles mechanical execution and quantitative validation. Position ToyzeroPlus as an AI-forward innovator in the traditional toy industry.

---

## Target Users

### Primary User Segment: Creative Directors at ToyzeroPlus

**Demographic Profile:**
- Role: Creative Director / Product Manager at character IP companies
- Experience: 5-10 years in toy design and product development
- Team size: Manages 2-5 designers and coordinates with production teams
- Location: Hong Kong / Asia-Pacific region

**Current Behaviors and Workflows:**
- Reviews 20-30 design concepts weekly during seasonal planning (Q1: Spring/CNY, Q2: Summer, Q3: Fall/Halloween, Q4: Winter/Christmas)
- Conducts manual competitive analysis via Instagram, trade shows, and retail visits
- Makes production volume decisions based on previous season sales and intuition
- Coordinates between design team, production, and sales with 4-6 week lead times

**Specific Needs and Pain Points:**
- Needs rapid design exploration that maintains brand consistency
- Struggles to quantify "gut feeling" about trend viability
- Requires data to justify production volumes to stakeholders
- Wants to free designers from repetitive seasonal variation work

**Goals They're Trying to Achieve:**
- Reduce concept-to-production cycle from 6 weeks to 2-3 weeks
- Increase confidence in production volume decisions (reduce 20-30% waste)
- Generate 3-4x more design variations for internal review without adding headcount
- Position company as innovation leader to attract investors and talent

---

## Goals & Success Metrics

### Business Objectives

- **Reduce Design Iteration Time by 70%:** From 3-4 weeks manual process to 3-5 hour AI-assisted workflow (Measured by: time tracking before/after implementation)

- **Improve Demand Forecast Accuracy by 30%:** Achieve R² > 0.7 on historical data simulation, reducing inventory mismatch costs (Measured by: LSTM model evaluation metrics)

- **Generate 4x Design Variations in Same Timeline:** Produce 16 seasonal variations (4 themes × 4 designs) vs. current 4 manual designs per season (Measured by: design output volume)

- **Complete FYP with Comprehensive Documentation:** Deliver working system, demo video, and 12,000-15,000 word report within 17-18 days (Measured by: project completion checklist)

### User Success Metrics

- **Creative Director Productivity:** Time saved per seasonal planning cycle (target: 15-20 hours saved)
- **Design Consistency Score:** CLIP similarity between generated variations and brand guidelines (target: >0.75 core, >0.60 style)
- **Forecast Usability:** Creative directors can interpret LSTM predictions without data science expertise (target: 5/5 usability rating)
- **Mood Board Quality:** Generated designs inspire final decisions vs. being discarded (target: >60% incorporation rate in final designs)

### Key Performance Indicators (KPIs)

- **NLP Prompt Quality (Obj 1):** Generate 28 viable SDXL prompts (7 themes × 4 variations) passing human review (Target: 100% acceptance rate)

- **LoRA Consistency (Obj 2):** Maintain character recognition across all generated variations (Target: CLIP core >0.75, style >0.60 on 100% of outputs)

- **LSTM Accuracy (Obj 3):** Predict sales within acceptable error margin (Target: MAE < 200 units, RMSE < 250 units, R² > 0.7 on test set)

- **System Integration (Obj 4):** End-to-end pipeline from trend input to forecast output completes without manual intervention (Target: <5 minutes execution time, 100% success rate on 3 test scenarios)

- **Academic Rigor (FYP Requirement):** Comparative experiments documented for LoRA (rank 8 vs 16) and LSTM (vs GRU), minimum 10 A/B test pairs (Target: Complete experimental report with statistical analysis)

---

## MVP Scope

### Core Features (Must Have)

**1. Google Trends Integration & Keyword Extraction**
- **Description:** Automated trend monitoring using pytrends library
- **Rationale:** Foundation for market-informed design; free and reliable data source
- **Deliverables:** `trends_extractor.py`, CSV exports of seasonal trends

**2. LLM-Based Prompt Generation**
- **Description:** GPT_API_free transforms keywords into detailed SDXL prompts maintaining character consistency
- **Rationale:** Bridges gap between raw trend data and actionable design specifications
- **Deliverables:** `prompt_generator.py`, template system, 28 example prompts

**3. Midjourney API Character Design Generation**
- **Description:** Integrate TTAPI Midjourney API with character reference (--cref) for consistent variations
- **Rationale:** Industry-standard tool used by ToyzeroPlus in production; high-quality outputs; character consistency via cref parameter; commercial viability
- **Deliverables:** `midjourney_generator.py`, TTAPI integration, 28 generated design variations (7 themes × 4 images), reference image selection documentation

**4. CLIP-Based Consistency Validation**
- **Description:** Automated similarity scoring against brand guidelines
- **Rationale:** Quantifies "brand consistency" objectively, replaces subjective design reviews
- **Deliverables:** `consistency_validator.py`, threshold configuration (>0.75/0.60)

**5. Hybrid LSTM Demand Forecasting**
- **Description:** Time-series (Google Trends) + static features (CLIP embeddings) model
- **Rationale:** Novel architecture addressing real-world constraint (new designs each season)
- **Deliverables:** `hybrid_lstm_model.py`, 60-point simulated dataset, trained weights

**6. Rule-Based Sales Simulation**
- **Description:** Generate 60 historical data points using Trends + CLIP + production limits + seasonal factors
- **Rationale:** Enables model training without 5 years of waiting; validates with ToyzeroPlus logic
- **Deliverables:** `simulate_sales.py`, simulation parameters documentation

**7. Streamlit Web Application**
- **Description:** Two-page interface (Design Generation + Forecast Dashboard)
- **Rationale:** Unified access point for non-technical users; demonstrates end-to-end integration
- **Deliverables:** Deployed app, user guide, 3 test scenarios documented

**8. TTAPI Midjourney API Integration**
- **Description:** Production-grade image generation via TTAPI platform with fast/turbo mode
- **Rationale:** Commercial-grade quality; industry validation (ToyzeroPlus uses Midjourney); character consistency via cref parameter; cost-effective (~$10-30 total)
- **Deliverables:** `ttapi_integration.py`, API key configuration guide, cost analysis documentation

### Out of Scope for MVP

- ❌ Scheduled/automated execution (cron jobs, GitHub Actions)
- ❌ User voting/feedback features
- ❌ Multi-character IP support (focus on single character)
- ❌ Real-time trend monitoring (batch processing only)
- ❌ Advanced hyperparameter tuning (use baseline configurations)
- ❌ Production deployment (local/cloud demo only)
- ❌ Extensive NLP model comparison (TextRank, RAKE benchmarks)
- ❌ Mobile app / responsive design optimization
- ❌ User authentication / multi-user support
- ❌ Export to production-ready formats (PSD, AI files)

### MVP Success Criteria

**The MVP is successful if:**

1. **Technical Validation:** All 4 objectives complete and demonstrable via 5-minute video
2. **Commercial Viability:** System uses industry-standard tools (Midjourney API) proving immediate business applicability
3. **End-to-End Flow:** User can input "春節, 紅色, 喜慶" → receive 4 mood boards + sales forecast in <5 minutes
4. **ToyzeroPlus Validation:** Creative director confirms system outputs match their current Midjourney workflow quality and are usable for real decision-making
5. **FYP Requirements Met:** 12,000-15,000 word report, demo video, code repository with documentation

**Acceptance Testing Scenarios:**

- Scenario A: Generate Spring Festival designs → Predict sales for Q1 2026
- Scenario B: Generate Halloween designs → Predict sales for Q4 2025
- Scenario C: Generate Christmas designs → Predict sales for Q4 2025

---

## Post-MVP Vision

### Phase 2 Features

**Scheduled Automation (Priority 1):**
- GitHub Actions / APScheduler for weekly trend monitoring
- Automated design generation every Monday
- Email/Slack notifications with new mood boards

**Multi-Character Support (Priority 2):**
- LoRA management system for ToyzeroPlus's full character lineup (5-8 characters)
- Character selector UI
- Comparative sales forecasts across characters

**Customer Voting Integration (Priority 3):**
- Public-facing voting portal for fans
- Integrate vote counts as LSTM input feature
- A/B testing framework for design variants

**Enhanced Forecasting (Priority 4):**
- Feature importance analysis dashboard
- "What-if" scenario modeling (e.g., "What if we increase production by 20%?")
- Regional trend differentiation (Hong Kong vs. Mainland China)

### Long-Term Vision (1-2 Years)

Transform the system from single-company tool to **industry-wide SaaS platform:**

- **IP Design Studio:** Multi-tenant platform serving 20-50 toy companies
- **Marketplace Integration:** Connect forecasts to manufacturing and distribution partners
- **AI Design Agents:** Autonomous systems that propose seasonal collections without human prompting
- **Reverse Engineering:** Input target sales goals → recommend optimal design features
- **Full IP Lifecycle Management:** From initial concept through production, marketing, and post-launch analytics

### Expansion Opportunities

1. **Adjacent Industries:**
   - Fashion accessories (character-themed bags, clothing)
   - Stationery and school supplies
   - Digital collectibles and NFTs

2. **Geographic Expansion:**
   - Japanese kawaii character market
   - Korean character goods industry
   - Western licensing (Disney, Sanrio model)

3. **Revenue Models:**
   - SaaS subscription ($500-2000/month per company)
   - Per-design generation fees ($5-20 per design)
   - Consulting services for AI adoption
   - White-label solutions for large enterprises

---

## Technical Considerations

### Platform Requirements

- **Target Platforms:** Web-based (Streamlit), accessible via modern browsers (Chrome, Safari, Edge)
- **Browser/OS Support:** No specific constraints; Streamlit handles cross-platform compatibility
- **Performance Requirements:**
  - Trend extraction: <30 seconds
  - Prompt generation: <10 seconds per prompt (40 seconds for 4)
  - Image generation via TTAPI Midjourney (fast mode): 60-90 seconds per image (4-6 minutes for 4 images)
  - LSTM inference: <5 seconds
  - Total end-to-end: <7 minutes acceptable (demo via video eliminates wait-time concern)

### Technology Preferences

**Frontend:**
- **Primary:** Streamlit (Python full-stack, 2-3 day development)
- **Backup:** Next.js 16 + AG Chat (if Streamlit proves insufficient)
- **Rationale:** Streamlit enables rapid prototyping without frontend expertise; sufficient for FYP demo

**Backend:**
- **Primary:** Python 3.9+ with modular architecture (`obj1_nlp_prompt/`, `obj2_midjourney_api/`, etc.)
- **APIs:** GPT_API_free (LLM), TTAPI Midjourney API (image generation)
- **Rationale:** Python ecosystem dominates ML/AI libraries; modular structure aligns with 4 objectives; Midjourney API provides commercial-grade image quality

**Database:**
- **Primary:** File-based storage (CSV for trends, NPY for embeddings, JSON for configurations)
- **Rationale:** Avoids database setup overhead; sufficient for MVP scale (60 data points, 28 designs)

**Hosting/Infrastructure:**
- **Primary:** Streamlit Cloud (free tier) for web app deployment
- **Generation:** TTAPI Midjourney API (PPU mode, ~$10-30 total cost)
- **Rationale:** Minimal infrastructure costs; aligns with commercial viability goals; Midjourney provides production-quality outputs

### Architecture Considerations

**Repository Structure:**
- **Decision:** Monorepo
- **Rationale:** Single FYP project; simplified dependency management; easier to package for submission

**Service Architecture:**
- **Decision:** Modular monolith (4 independent modules within single codebase)
- **Structure:**
  ```
  src/
  ├── obj1_nlp_prompt/       # Independent: Trends → Prompts
  ├── obj2_midjourney_api/   # Independent: Prompts → Midjourney Images
  ├── obj3_lstm_forecast/    # Depends on: Obj 2 CLIP outputs
  └── obj4_web_app/          # Integrates: All objectives
  ```
- **Rationale:** Balances modularity (easier testing/debugging) with simplicity (no microservice overhead); commercial-ready architecture

**Integration Requirements:**
- **Data Flow:** Obj 1 outputs → Obj 2 inputs; Obj 2 CLIP features → Obj 3 LSTM training; All modules → Obj 4 UI
- **API Dependencies:**
  - External: pytrends, GPT_API_free, TTAPI Midjourney API, CLIP (via transformers)
  - Internal: Python function calls between modules
- **File System Contracts:**
  - `data/trends/*.csv` - Obj 1 outputs
  - `data/clip_embeddings/*.npy` - Obj 2 outputs
  - `data/generated_images/*.png` - Obj 2 Midjourney outputs
  - `models/lstm/*.pth` - Obj 3 weights

**Security/Compliance:**
- **API Key Management:** Environment variables (.env file, not committed to Git)
- **Data Privacy:** Open-source Pikachu images used for FYP demonstration; no proprietary data uploaded
- **Licensing:** All dependencies use permissive licenses (MIT, Apache 2.0); Midjourney API usage complies with TTAPI terms of service

---

## Constraints & Assumptions

### Constraints

**Budget:**
- **Limit:** ~$10-30 for TTAPI Midjourney quota (28-40 image generations)
- **Rationale:** Commercial viability demonstration—proves system is economically feasible for real businesses; minimal cost validates production readiness
- **Implications:** No expensive enterprise APIs; focus on cost-effective commercial tools that ToyzeroPlus can actually afford

**Timeline:**
- **Hard Limit:** 17-18 days (2.5 weeks) from start to FYP submission
- **Breakdown:**
  - Day 1-3: Objective 1 (NLP)
  - Day 4-5: Objective 2 (Midjourney API Integration) - **Saves 2 days vs LoRA training**
  - Day 6-9: Objective 3 (LSTM)
  - Day 10-12: Objective 4 (Web)
  - Day 13-15: Testing + Demo + Documentation
  - Day 16-18: Buffer + Polish
- **Implications:** Faster delivery timeline due to Midjourney API vs custom model training; allows more time for polish and documentation

**Resources:**
- **Personnel:** Solo developer (FYP student)
- **Compute:** TTAPI Midjourney API (PPU mode, ~$10-30 total)
- **Data:** 1-2 high-quality Pikachu reference images (freely available online), simulated historical sales data
- **Implications:** No compute-intensive training; API-based approach enables rapid iteration and commercial-grade quality

**Technical:**
- **Language:** Python 3.9+ (all code must be Python for consistency)
- **Platform:** Must run on standard laptop (MacBook / Windows) without specialized hardware
- **Dependencies:** Only open-source libraries with active maintenance; commercial APIs with stable terms of service
- **Implications:** Focus on production-ready tools (Midjourney API) rather than experimental models

### Key Assumptions

- **Pikachu Reference Images:** 1-2 high-quality Pikachu images available online for Midjourney cref parameter
- **TTAPI Stability:** TTAPI Midjourney API remains accessible and stable throughout 17-18 day development period; quota pricing remains at estimated $10-30 range
- **API Stability:** GPT_API_free remains accessible throughout development
- **Midjourney Character Consistency:** Midjourney's --cref (character reference) feature provides sufficient consistency for character IP variations (industry validation: ToyzeroPlus uses Midjourney for this exact purpose)
- **Trend Data Validity:** Google Trends in Traditional Chinese (Hong Kong) accurately represents toy industry target market
- **LSTM Feasibility:** 60 simulated data points provide sufficient signal for model training (academic consensus: 50-100 points minimum for LSTM)
- **Design Acceptance:** Mood board quality threshold acceptable at 60% designer incorporation rate (vs. 100% polished finals)
- **Demo Format:** Video demo (5-6 minutes) is acceptable for FYP evaluation; live system not required during presentation (confirmed by user)
- **Simulation Validity:** Self-generated rule-based sales simulation (Trends + CLIP + constraints) provides sufficient training data without requiring real historical sales data (confirmed by user)
- **Commercial Positioning:** FYP evaluation accepts commercial-focused projects demonstrating real business value rather than purely academic research

---

## Risks & Open Questions

### Key Risks

**Risk 1: TTAPI Service Reliability & Pricing Changes**
- **Description:** TTAPI Midjourney API downtime, rate limiting, or unexpected quota pricing changes during development
- **Impact:** MEDIUM - Blocks Objective 2 but backup exists
- **Mitigation:**
  - Pre-purchase TTAPI quota at start (Day 0-1) to lock in pricing
  - Test API throughput limits on Day 4 before committing to full generation
  - Backup plan: Use DALL-E 3 API or Flux API (also available on TTAPI) if Midjourney fails
  - Cache strategy: Generate and store all 28 images early; web app reads from local cache

**Risk 2: Midjourney Character Consistency Quality**
- **Description:** Midjourney cref parameter doesn't maintain sufficient character consistency across variations
- **Impact:** MEDIUM - Core value proposition (brand consistency) fails
- **Mitigation:**
  - Early validation: Generate 4-5 test images on Day 4 to verify cref quality
  - Reference image selection: Test multiple Pikachu reference images to find optimal one
  - Prompt engineering: Refine prompts to emphasize character features ("yellow electric mouse, red cheeks, brown stripes on back")
  - CLIP validation: Use >0.70 threshold (lower than 0.75) if cref proves less consistent than expected
  - Fallback acceptance: Acknowledge limitation in FYP report if consistency isn't perfect—focus on commercial workflow automation value

**Risk 3: Free API Reliability (GPT_API_free)**
- **Description:** Rate limits, downtime, or service discontinuation during development
- **Impact:** LOW-MEDIUM - Blocks Objective 1 but alternatives exist
- **Mitigation:**
  - GPT_API_free backup: Hugging Face Mistral-7B (free inference API)
  - Pre-generate outputs: Store all 28 prompts locally on Day 3; web app reads from cache
  - Manual fallback: Write prompts manually if both APIs fail (28 prompts = 2-3 hours work)

**Risk 4: LSTM Underfitting (Insufficient Data)**
- **Description:** 60 simulated data points may be too few for LSTM to learn meaningful patterns
- **Impact:** MEDIUM - Hybrid LSTM forecasting fails to demonstrate value
- **Mitigation:**
  - GRU backup: Simpler architecture may perform better with limited data
  - Extend simulation: Generate 120 points (10 years) if time permits during Obj 3
  - Adjust success criteria: R² > 0.6 acceptable for academic demonstration if dataset limitation is acknowledged
  - Fallback positioning: Emphasize proof-of-concept value and commercial workflow rather than forecast accuracy

**Risk 5: Streamlit Performance/Capability Limits**
- **Description:** Streamlit cannot handle LSTM inference or TTAPI calls efficiently
- **Impact:** LOW - Obj 4 delayed but backup exists
- **Mitigation:**
  - Pre-testing: Validate Streamlit + LSTM + TTAPI integration in Day 10 first 2 hours
  - Next.js backup: Pivot to Next.js 16 + AG Chat if Streamlit fails (adds 1 day to timeline)
  - Cached demo: Pre-generate all outputs; Streamlit only displays cached results (eliminates API calls during demo)

**Risk 6: Time Overrun (Objective Scope Creep)**
- **Description:** Individual objectives exceed allocated time (e.g., Obj 2 takes 4 days instead of 2)
- **Impact:** HIGH - Cascades to all subsequent objectives; jeopardizes FYP submission
- **Mitigation:**
  - Strict time boxing: End-of-day checkpoint; if behind, cut scope immediately
  - Pre-defined cuts: Documented "nice-to-have" features to drop first (e.g., prompt optimization, LSTM vs GRU comparison)
  - Buffer days: Day 16-18 absorbs 2-3 day overrun; if exceeded, reduce testing/documentation quality

### Open Questions

1. **TTAPI Quota Pricing:**
   - **Question:** What is the exact quota pricing for TTAPI Midjourney API? (Estimated $10-30 but needs confirmation)
   - **Action:** Register TTAPI account and check pricing page before committing to purchase

2. **Commercial Positioning:**
   - ✅ **RESOLVED:** FYP can focus on commercial viability and production-readiness rather than pure academic research
   - ✅ **RESOLVED:** ToyzeroPlus uses Midjourney in production, validating commercial approach

3. **Technical Validation:**
   - What constitutes "acceptable" forecast accuracy for business utility? - **Assumed:** R² > 0.7, MAE < 200 units, RMSE < 250 units as per MVP Success Criteria
   - Should LSTM predict absolute sales numbers or relative rankings? - **Assumed:** Absolute sales numbers (more useful for production planning)
   - How to validate CLIP similarity thresholds (0.75/0.60)? - **Assumed:** Thresholds based on academic literature; may lower to 0.70 if Midjourney cref proves less consistent

4. **Scope Boundary:**
   - If major technical blocker emerges, is pivoting acceptable? - **Assumed:** Yes, backup plans documented (DALL-E 3/Flux for Midjourney, GRU for LSTM)
   - Should web app include error handling? - **Assumed:** Basic error handling for demo robustness; comprehensive handling out of scope

5. **Post-FYP Considerations:**
   - Continue development post-FYP for commercial deployment? - **Assumed:** System designed for immediate commercial viability; ToyzeroPlus can adopt after FYP
   - Code quality focus? - **Assumed:** Balance clean code with rapid development; production-ready architecture for potential commercialization

### Areas Needing Further Research

- **TTAPI Midjourney Pricing:** Exact quota pricing model and rate limits for PPU mode
- **Midjourney cref Effectiveness:** Character reference parameter quality testing with different reference images
- **LSTM Architecture Details:** Optimal number of layers (1, 2, or 3?), hidden dimensions (64, 128, 256?), dropout rates (0.1, 0.2, 0.3?)
- **Simulation Parameter Tuning:** What weights for Google Trends (30%), CLIP similarity (25%), seasonal factors (20%) best approximate real-world dynamics?
- **Trend Keyword Selection:** Which Google Trends categories (Shopping, Entertainment, Arts & Entertainment) most relevant for toy industry?
- **Web App UX:** Wireframe needed for optimal information density vs. simplicity trade-off in Streamlit interface
- **Prompt Engineering:** Optimal prompt structure for Midjourney cref to maintain character consistency while incorporating seasonal trends

---

## Appendices

### A. Research Summary

**Market Research Findings:**
- ToyzeroPlus operates in Hong Kong character goods market (estimated $500M annually)
- Seasonal product cycles: 4 major seasons (Spring/CNY, Summer, Fall/Halloween, Winter/Christmas)
- Production lead times: 6-8 weeks from design freeze to retail
- Inventory risk: 20-30% unsold stock typical for new seasonal designs

**Competitive Analysis:**
- **Traditional Workflow:** Manual design (Adobe CC) + gut-feel forecasting (Excel)
- **Current ToyzeroPlus Workflow:** Midjourney for design inspiration but lacks integration with trend analysis and demand forecasting
- **Enterprise Forecasting:** SAP/Oracle too expensive and complex for mid-sized toy companies
- **Academic Research:** Hybrid LSTM architectures applied to fashion forecasting (analogous domain) but not toy industry; no integrated systems combining trend monitoring, AI generation, and forecasting

**Technical Feasibility Studies:**
- **Midjourney Character Reference (cref):** Midjourney v6+ supports --cref parameter for character consistency; validated by ToyzeroPlus production use
- **TTAPI Commercial Availability:** TTAPI provides stable PPU mode for Midjourney API access with reasonable pricing (~$10-30 estimated for FYP scope)
- **LSTM for Limited Data:** 50-100 time points considered minimum viable for LSTM training (academic consensus)
- **Free Resource Availability:** Validated pytrends (stable), GPT_API_free (unlimited ChatGPT access), TTAPI Midjourney (pay-per-use commercial API)

### B. Stakeholder Input

**ToyzeroPlus Creative Director (Informal Feedback):**
- Validated market need for trend-informed design and demand forecasting
- **Confirmed active use of Midjourney for character design exploration** (validates commercial viability of AI-assisted design)
- Expressed high interest in demand forecasting (current intuition-based approach causes stress)
- Confirmed "mood board generator" positioning appropriate for industry adoption
- System will use open-source Pikachu character for FYP demonstration to showcase methodology applicable to any IP without compromising proprietary assets

**FYP Supervisor Guidance:**
- Emphasized commercial value and production-readiness as acceptable FYP focus (not purely academic research required)
- Recommended video demo format to avoid live presentation technical issues
- Suggested 12,000-15,000 word report as target length (standard for engineering FYP)
- Encouraged using industry-standard tools (like Midjourney) to demonstrate real-world applicability

### C. References

**Technical Documentation:**
- Midjourney Documentation: https://docs.midjourney.com/
- TTAPI Midjourney API: https://ttapi.io/docs/apiReference/midjourney
- CLIP: https://arxiv.org/abs/2103.00020
- LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**Key Libraries:**
- pytrends: https://pypi.org/project/pytrends/
- TTAPI Python SDK: https://ttapi.io/docs
- PyTorch: https://pytorch.org/
- Streamlit: https://docs.streamlit.io/
- Transformers (CLIP): https://huggingface.co/docs/transformers/

**Related Projects:**
- GPT_API_free: https://github.com/chatanywhere/GPT_API_free
- TTAPI Platform: https://ttapi.io

---

## Next Steps

### Immediate Actions

1. **Register TTAPI Account & Purchase Quota** (Day 0 - PRIORITY)
   - Register at https://ttapi.io
   - Check exact quota pricing for Midjourney API (PPU mode)
   - Purchase initial quota (~$10-30 estimated) to lock in pricing
   - Generate API key and test basic Midjourney API call
   - Document pricing structure for cost analysis in FYP report

2. **Select Pikachu Reference Images** (Day 0)
   - Search official Pokémon sources, DeviantArt, Pinterest for high-quality Pikachu images
   - Target: 1-2 reference images (high resolution, clear character features)
   - Test multiple images for Midjourney cref parameter effectiveness
   - Upload to publicly accessible URL (required for --cref parameter)
   - Download and organize in `data/reference_images/` directory

3. **Register All Free Service Accounts** (Day 0)
   - GPT_API_free: https://github.com/chatanywhere/GPT_API_free
   - Hugging Face: Create account for CLIP model access
   - Streamlit Cloud: Register for deployment

4. **Set Up Development Environment** (Day 0)
   - Install Python 3.9+, pip, Git
   - Create project repository structure (src/, data/, models/, docs/)
   - Install base dependencies: pytrends, openai, torch, transformers, streamlit, requests (for TTAPI)
   - Create .env template for API keys (TTAPI_KEY, GPT_API_KEY)

5. **Validate Critical Path Assumptions** (Day 0-1)
   - Test TTAPI Midjourney API call with cref parameter (confirm character consistency works)
   - Test GPT_API_free API call (confirm unlimited access)
   - Test pytrends data extraction (confirm Traditional Chinese support)
   - Document test results for Day 1 start

### PM Handoff

This Project Brief provides the full context for **AI-Driven Market-Informed Character IP Design Extension and Demand Forecasting System**.

**For PRD Generation:**
Please start in 'PRD Generation Mode', review the brief thoroughly to work with the user to create the PRD section by section as the template indicates, asking for any necessary clarification or suggesting improvements.

**Key Focus Areas for PRD:**
- Translate 4 objectives into structured Functional Requirements (FRs)
- Define Non-Functional Requirements (NFRs) for performance, testing, documentation
- Break down into Epics following logical sequence:
  - Epic 1: Foundation + Objective 1 (NLP Pipeline)
  - Epic 2: Objective 2 (LoRA Training + CLIP Validation)
  - Epic 3: Objective 3 (LSTM Forecasting + Simulation)
  - Epic 4: Objective 4 (Web Integration + Demo)
  - Epic 5: Testing, Documentation, FYP Delivery
- Ensure Stories within each Epic are sized for single AI agent execution (2-4 hours)
- Incorporate technical constraints (free resources, 17-18 day timeline, Python-only)

**Assumptions to Validate with User:**
- Confirm ToyzeroPlus image acquisition plan
- Confirm acceptable LSTM accuracy thresholds (R² > 0.7?)
- Confirm FYP documentation requirements (12k-15k words, video demo format)
- Clarify priority if timeline slips (cut scope vs. extend deadline)

---

**Document Status:** ✅ Draft Complete - Ready for User Review and PRD Generation

**Estimated Review Time:** 15-20 minutes
**Next Milestone:** PRD Section-by-Section Creation
