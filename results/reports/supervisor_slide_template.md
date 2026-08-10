# Supervisor Meeting Presentation Template

## Presentation Goal

Create a clear, simple, academic slide presentation for a research meeting with my supervisor.

The purpose is **not** to reproduce the full report.  
The slides should help me explain:

- what I worked on
- what changed since the previous meeting
- what experiments I ran
- what worked and what did not
- the most important results
- what the results mean
- what decisions or feedback I need from my supervisor
- what I should do next

---

## Design Style

Use a clean academic style.

### Visual rules

- Simple and professional
- White or very light background
- Minimal decoration
- No flashy gradients or unnecessary graphics
- Consistent typography
- Strong slide titles
- Plenty of whitespace
- Short bullets
- Avoid dense paragraphs
- Avoid putting the whole report on the slides
- Use charts, diagrams, tables, or visual comparisons when they communicate the result better than text
- Highlight only the most important numbers
- Use bold selectively
- Keep tables small and readable
- Prefer one main message per slide

### Tone

The presentation should feel like a **research discussion with a supervisor**, not a conference keynote and not a thesis defense.

It should be easy to interrupt, discuss, and ask questions during the presentation.

---

## Recommended Length

Default: **8–15 slides**

Adjust depending on the amount of material.

Do not create unnecessary slides just to reach a specific number.

---

# Recommended Slide Structure

## Slide 1 — Meeting Topic

Include:

- short project or experiment title
- brief subtitle if useful
- my name
- meeting context or date if appropriate

Keep this slide minimal.

---

## Slide 2 — Research Question / Objective

Explain the main question in 1–3 short points.

For example:

- What are we trying to predict or understand?
- Why is this experiment needed?
- What comparison are we testing?

If the project already has established objectives, summarize only the objectives relevant to this meeting.

---

## Slide 3 — Current Study Setup

Show the experimental setup visually where possible.

Include only the information needed to understand the results, such as:

- dataset
- spatial unit
- temporal period
- train / validation / test split
- prediction target
- number of samples
- number of positive cases
- main feature groups

Prefer a simple workflow diagram or compact summary over long text.

---

## Slide 4 — Method / Model Setup

Show the main models or experimental branches being compared.

Examples:

- baseline models
- GNN model
- hybrid model
- transfer learning model
- feature augmentation
- sampling strategy

Use a diagram when model relationships are easier to understand visually.

Do not include implementation details unless they affect interpretation.

---

## Slide 5 — Evaluation Strategy

Explain how performance is evaluated.

Focus on metrics that actually matter for the research question.

For rare-event prediction, prioritize metrics such as:

- Average Precision / PR-AUC
- Recall@K
- Precision@K
- ROC-AUC only as supporting information
- calibration metrics when relevant

Explain briefly why the primary metric matters.

---

## Slide 6 — Main Result

This should usually be one of the most important slides.

Show the primary model comparison.

Prefer:

- bar chart
- dot plot
- compact comparison table

Highlight:

- best model
- strongest baseline
- practical difference
- key metric

Add one short takeaway sentence.

Example:

**Takeaway:** GraphSAGE produced the strongest rare-event ranking, while tree models retained higher ROC-AUC.

---

## Slide 7 — Important Experiment / Ablation

Show one important experiment that helps explain the main result.

Examples:

- negative sampling
- graph topology
- transfer learning
- feature removal
- architecture comparison
- hyperparameter experiment

Only include experiments that help answer a research question.

Add a clear interpretation below the result.

---

## Slide 8 — What Worked / What Did Not

Use a simple two-column layout if appropriate.

### Worked

Examples:

- spatial message passing improved ranking
- specific sampling strategy helped
- transfer learning rescued a weak architecture

### Did Not Improve Performance

Examples:

- embedding stacking
- multitask learning
- more complex graph architecture
- hybrid fusion

Do not hide negative results.  
Present them as useful evidence about what the model does and does not need.

---

## Slide 9 — Interpretation / Feature Importance

Show what the model appears to be learning.

Possible content:

- top predictive features
- risk-factor groups
- spatial patterns
- important environmental variables
- outbreak-history effects

Clearly distinguish:

**predictive association** from **causal effect**.

Prefer a small ranked chart rather than a large table.

---

## Slide 10 — Key Takeaways

Use 3–5 concise statements.

Each point should answer:

- What did we learn?
- What is the strongest result?
- What should we stop pursuing?
- What is still uncertain?

Avoid repeating the results table.

Focus on interpretation.

---

## Final Slide — Discussion With Supervisor

This slide is important.

Instead of ending with only "Thank you", include 3–5 questions or decisions for discussion.

Examples:

- Is GraphSAGE performance strong enough to become the main model?
- Should the paper emphasize ranking performance rather than probability prediction?
- Is the wild-bird transfer experiment worth keeping in the manuscript?
- Should we prioritize improving the graph structure or improving the features?
- What experiment is still necessary before writing the paper?
- Should we simplify the model comparison section?
- Which result should become the main contribution?

The purpose is to turn the presentation into a productive research discussion.

---

# Content Selection Rules

When converting a report, Markdown file, paper draft, experiment log, or analysis into slides:

1. Read the entire source first.
2. Identify the research story.
3. Do not make one slide for every section.
4. Remove details that do not affect the meeting discussion.
5. Prioritize results over background.
6. Keep methodology only detailed enough to understand the results.
7. Show failed experiments if they affect the next research decision.
8. Combine related experiments when possible.
9. Highlight changes since the previous version if they are known.
10. End with concrete questions for the supervisor.

---

# Table Rules

Avoid copying large tables directly from the report.

If a table has many rows:

- show only the most relevant models
- highlight the best result
- move secondary models into a smaller supporting table
- use a chart if comparison is easier visually

Large appendix-style tables should not appear in the main meeting deck unless necessary.

---

# Chart Rules

When possible:

- directly label important values
- order models from best to worst
- visually emphasize the main model
- avoid unnecessary legends
- avoid 3D charts
- avoid overly complicated multi-panel figures
- make axis labels readable
- state the takeaway on the same slide

---

# Writing Style for Slides

Use short, natural academic language.

Prefer:

**GraphSAGE achieved the highest AP on the 2025 test set.**

Instead of:

**It can clearly be observed from the obtained experimental results that the GraphSAGE model demonstrated superior performance.**

Prefer:

**Wild-bird pretraining improved R-GCN substantially, but did not outperform GraphSAGE.**

Instead of long explanations.

---

# Default Output Request

When I provide research content together with this file, create an editable PowerPoint presentation.

Default requirements:

- 8–15 slides
- clean supervisor-meeting style
- simple academic design
- minimal text
- clear visual hierarchy
- charts or diagrams where useful
- preserve the actual results and numbers from my source
- do not invent results
- do not make claims stronger than the evidence supports
- include negative results when scientifically relevant
- finish with discussion questions / next decisions

Before finalizing the deck, make sure the presentation tells a coherent story:

**Problem → Setup → Experiment → Result → Interpretation → Decision / Next Step**
