# System Architecture

## Overview

The platform is structured as a layered analytics system where each layer is isolated, testable, and reusable.

## Layers

1. Presentation Layer
   - Streamlit UI
   - Sidebar navigation
   - KPI cards and tabs
   - Download and report actions

2. Data Ingestion Layer
   - CSV and Excel input
   - Sample dataset generation
   - SQLite or PostgreSQL persistence for experiment metadata and results

3. Validation and Cleaning Layer
   - Required column checks
   - Missing value detection
   - Duplicate detection
   - Group validation and domain checks

4. Analytics Layer
   - KPI calculations
   - Hypothesis testing
   - Confidence intervals
   - Effect size
   - Power and MDE estimation
   - Segmentation analysis
   - Guardrail analysis

5. Decision Layer
   - Rule-based engine
   - Recommendation generation
   - Business impact interpretation

6. Reporting Layer
   - CSV export
   - HTML summary export
   - Analyst-ready recommendation text

## Data Flow

User -> Upload/Generate Data -> Validate -> Clean/Reject -> Analyze -> Visualize -> Recommend -> Export

## Database Design

The database schema stores:

- experiments
- experiment_runs
- analysis_results

SQLite is the default local backend. Setting the `DATABASE_URL` environment variable selects PostgreSQL without changing the analytical pipeline.

## Design Principles

- Separation of concerns
- Statistical correctness over dashboard aesthetics
- User-friendly interpretation for non-technical readers
- Transparent assumptions and uncertainty
- Modular testing before deployment
