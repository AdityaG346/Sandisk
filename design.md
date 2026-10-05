# DieYield Intelligence --- SanDisk-Inspired Design System

**Version:** 1.0\
**Purpose:** Judge-facing visual and interaction system for the
Multi-Resolution Die Yield Prediction application\
**Design direction:** SanDisk-inspired industrial technology /
semiconductor intelligence\
**Status:** Design specification --- UI/UX only; no ML/model behavior is
defined here.

------------------------------------------------------------------------

## 1. Design Intent

The application should feel like a **real semiconductor
yield-intelligence product**, not a hackathon dashboard or a generic
Streamlit analytics page.

The visual direction is derived from:

-   the current SanDisk corporate website screenshots supplied for this
    project;
-   SanDisk's current brand direction and public design statements;
-   the project's existing technical-dashboard functionality.

SanDisk's current brand direction describes a **"Mindset of Motion"**,
with a visual language built around a pixel origin, bold geometry, clean
lines, openness, and future-facing technology. SanDisk also describes
its newer red as intentionally warmer and more immediate. The new mark
is explicitly described as pixel-driven, angular, minimalist, and based
on broad geometric forms. These principles should influence the product
without copying the consumer website literally.

### Product personality

The application should communicate:

1.  **Precision**
2.  **Engineering confidence**
3.  **Speed**
4.  **Data intelligence**
5.  **Industrial reliability**
6.  **Modernity**
7.  **Clarity under pressure**

### Avoid

-   student-project styling;
-   excessive emojis;
-   rainbow dashboards;
-   excessive gradients;
-   excessive glassmorphism;
-   decorative elements that do not explain data;
-   generic SaaS purple/blue branding;
-   fake "factory control" aesthetics;
-   unnecessary animation;
-   overly rounded consumer-app cards.

------------------------------------------------------------------------

# 2. Brand Relationship

The product is **SanDisk-inspired**, not an attempt to impersonate the
official SanDisk website.

Use SanDisk's recognizable visual principles:

-   warm red;
-   black / near-black;
-   white;
-   strong geometric typography;
-   pixel / modular motifs;
-   angular separators;
-   bold, high-contrast hero numbers;
-   minimal but purposeful motion.

Do **not** recreate the SanDisk website navigation, product catalog,
legal footer, shopping links, or corporate site structure.

This is a **technical yield-intelligence product**.

------------------------------------------------------------------------

# 3. Core Color System

## 3.1 Primary Brand Red

Use one primary red consistently.

### `--sd-red`

**#F51B0B**

Purpose:

-   primary brand accent;
-   selected controls;
-   active navigation;
-   important KPI highlights;
-   failure-state accents when the context is brand/interaction rather
    than ground-truth failure;
-   CTA buttons;
-   thin separators;
-   hover states.

### Red usage rule

Red should be **high-impact, not everywhere**.

Target approximately:

-   5--10% of visible interface area;
-   larger usage only in hero/active-state moments.

Do not make every card red.

------------------------------------------------------------------------

## 3.2 Deep Black

### `--sd-black`

**#050505**

Purpose:

-   page background;
-   hero sections;
-   high-contrast presentation areas.

------------------------------------------------------------------------

## 3.3 Surface Black

### `--sd-surface-1`

**#0D0F12**

Purpose:

-   primary dashboard surface;
-   sidebar background;
-   navigation regions.

------------------------------------------------------------------------

## 3.4 Surface Dark

### `--sd-surface-2`

**#15181D**

Purpose:

-   cards;
-   panels;
-   table containers;
-   chart backgrounds.

------------------------------------------------------------------------

## 3.5 Surface Elevated

### `--sd-surface-3`

**#1C2128**

Purpose:

-   selected cards;
-   hover states;
-   expanded sections;
-   elevated controls.

------------------------------------------------------------------------

## 3.6 Border

### `--sd-border`

**#30353D**

Purpose:

-   card borders;
-   table separators;
-   control boundaries.

Use 1px borders.

Avoid bright white borders.

------------------------------------------------------------------------

## 3.7 Primary Text

### `--sd-white`

**#FFFFFF**

Primary:

-   headings;
-   hero numbers;
-   selected navigation;
-   important labels.

------------------------------------------------------------------------

## 3.8 Secondary Text

### `--sd-text-secondary`

**#B7BDC7**

Purpose:

-   supporting descriptions;
-   metadata;
-   chart labels;
-   secondary navigation.

------------------------------------------------------------------------

## 3.9 Muted Text

### `--sd-text-muted`

**#7F8792**

Purpose:

-   footnotes;
-   helper text;
-   technical disclosures;
-   low-priority metadata.

------------------------------------------------------------------------

# 4. Semantic Data Colors

Brand red must NOT be used for every "bad" value because red is also a
brand color.

Use separate semantic colors.

## Success / Positive

### `--sd-success`

**#35D07F**

Use for:

-   verified;
-   improvement;
-   positive delta;
-   successful system state.

------------------------------------------------------------------------

## Warning

### `--sd-warning`

**#F4B740**

Use for:

-   caution;
-   borderline risk;
-   methodological warnings.

------------------------------------------------------------------------

## Failure / Ground Truth

### `--sd-failure`

**#FF5A5F**

Use for:

-   actual failed die;
-   ground-truth failure;
-   critical model miss.

------------------------------------------------------------------------

## Model B Accent

### `--sd-model-b`

**#A855F7**

Use sparingly to distinguish Model B from Model A.

Do not use purple as a general brand color.

------------------------------------------------------------------------

## Model A Accent

### `--sd-model-a`

**#25B9E6**

Use for:

-   Model A curve;
-   Model A benchmark;
-   baseline visual identity.

------------------------------------------------------------------------

## Spatial Context

### `--sd-context`

**#43D17C**

Use for:

-   spatial context;
-   non-interventionist contextual features.

Always pair with the label:

> Context --- not an intervention

------------------------------------------------------------------------

# 5. Color Hierarchy

The visual hierarchy should be:

1.  White --- information
2.  SanDisk red --- interaction / brand emphasis
3.  Cyan --- Model A
4.  Purple --- Model B
5.  Green --- positive/verified
6.  Amber --- caution
7.  Red/pink --- actual failure
8.  Muted gray --- supporting information

Do not use more than 4 strong colors simultaneously in one component.

------------------------------------------------------------------------

# 6. Typography

## 6.1 Primary UI Typeface

Preferred:

### Inter

Use:

-   `Inter`
-   fallback: `Arial`
-   fallback: `Helvetica`
-   fallback: sans-serif

The UI should feel modern and technical.

------------------------------------------------------------------------

## 6.2 Display Typography

Use a bold geometric sans-serif treatment.

Preferred:

-   Inter ExtraBold / Bold;
-   fallback Arial Bold.

Do not use decorative futuristic fonts.

The SanDisk brand mark itself has broad, geometric, slab-serif
characteristics, but application UI text should remain highly readable.

------------------------------------------------------------------------

# 7. Type Scale

## Display / Hero

### 48--64px

Weight: 700--800\
Line height: 0.95--1.05

Use only for:

-   primary product title;
-   hero metric;
-   major result.

------------------------------------------------------------------------

## Page Heading

### 32--40px

Weight: 700\
Line height: 1.1

Example:

> Multi-Resolution Die Yield Prediction

------------------------------------------------------------------------

## Section Heading

### 24--28px

Weight: 700\
Line height: 1.2

Example:

> Why Did Model B Change Its Mind?

------------------------------------------------------------------------

## Card Heading

### 16--18px

Weight: 650--700

------------------------------------------------------------------------

## Body

### 14--16px

Weight: 400--500\
Line height: 1.45--1.6

------------------------------------------------------------------------

## Metadata

### 11--13px

Weight: 500\
Letter spacing: 0.02--0.05em

Use for:

-   TEST SET;
-   MODEL STATUS;
-   calibration;
-   timestamps;
-   technical notes.

------------------------------------------------------------------------

## KPI Numbers

### 32--48px

Weight: 750--800

KPI numbers should be the most visually dominant element inside cards.

------------------------------------------------------------------------

# 8. Text Style Rules

Use sentence case for normal UI.

Avoid:

> MODEL PERFORMANCE ANALYSIS

Prefer:

> Model performance

Use uppercase only for small labels:

> TEST POPULATION

> MODEL STATUS

> INSPECTION BUDGET

Uppercase labels should be 10--12px with moderate letter spacing.

------------------------------------------------------------------------

# 9. Layout System

Use a 12-column conceptual grid.

### Desktop

-   max content width: 1440--1600px;
-   page horizontal padding: 32--48px;
-   sidebar width: 250--280px;
-   main content width: remaining viewport;
-   section spacing: 48--72px.

### Card spacing

-   internal padding: 20--28px;
-   card gap: 16--24px.

### Small components

-   gap: 8--12px.

------------------------------------------------------------------------

# 10. Border Radius

Avoid the overly rounded SaaS aesthetic.

### Cards

8--12px

### Buttons

6--8px

### Inputs

6--8px

### Pills / status badges

999px

Use rounded shapes only when they communicate status or selection.

------------------------------------------------------------------------

# 11. Borders and Shadows

Use subtle borders.

Default:

``` css
border: 1px solid #30353D;
```

Avoid large drop shadows.

For elevated cards:

``` text
0 8px 30px rgba(0,0,0,0.20)
```

The interface should feel engineered rather than floating.

------------------------------------------------------------------------

# 12. Pixel / Modular Design Motif

The new SanDisk identity is explicitly rooted in the pixel and modular
geometry.

Use a subtle pixel motif in:

-   section dividers;
-   background decoration;
-   loading states;
-   empty states;
-   architecture diagrams;
-   data-grid highlights.

Example:

``` text
■ ■ ■
  ■ ■
    ■
```

Keep it subtle.

Never cover the content with decorative pixel patterns.

------------------------------------------------------------------------

# 13. Angular Design Motif

Use small angular cuts or chamfers for:

-   active navigation indicators;
-   architecture cards;
-   hero metric accents;
-   section separators.

The effect should feel inspired by semiconductor geometry.

Do not turn every card into a polygon.

------------------------------------------------------------------------

# 14. Header

The application should have a clean product header.

### Recommended structure

``` text
DIEYIELD INTELLIGENCE
Multi-Resolution Yield Prediction

                    ● MODEL VERIFIED
```

Underneath:

``` text
40 HELD-OUT WAFERS   ·   32,598 ELIGIBLE DIES   ·   1,380 NEW FAILURES
```

This immediately establishes credibility.

------------------------------------------------------------------------

# 15. Sidebar

The sidebar should feel like a professional control center.

## Header

``` text
DIEYIELD
INTELLIGENCE
```

Small descriptor:

> Multi-Resolution Yield Prediction

------------------------------------------------------------------------

## Demo Mode

``` text
DEMO MODE

Demo Case
[ W_N_0066 · Die (2,13) ]
```

------------------------------------------------------------------------

## Wafer

``` text
WAFER

[ W_N_0066 ▼ ]
```

------------------------------------------------------------------------

## View

``` text
VIEW

○ Model B
○ Model A
● A vs B
```

------------------------------------------------------------------------

## Dataset

``` text
DATASET

40 held-out wafers
32,598 eligible dies
```

------------------------------------------------------------------------

## Model

``` text
MODEL

A  511 features
B  531 features

● Evaluation verified
```

Do NOT show:

> SanDisk Die Yield Prediction Hackathon Deliverable

Do not show hackathon-internal language anywhere in the product UI.

------------------------------------------------------------------------

# 16. Executive Results

This is the most important section.

Headline:

> **Model B captures more failures at the same inspection budget.**

Subheading:

> Adding sub-die evidence improves continuous risk ranking without
> increasing the inspection budget.

------------------------------------------------------------------------

## KPI 1

### PR-AUC

**0.5025 → 0.5362**

`Δ +0.0337`

Secondary:

> 95% wafer-cluster bootstrap CI \[+0.023, +0.043\]

------------------------------------------------------------------------

## KPI 2 --- HERO

### 10% Inspection Budget

**768 → 834 failures**

### **+66 additional failures captured**

Secondary:

> 55.7% → 60.4% failure capture

This should be the strongest KPI on the page.

------------------------------------------------------------------------

## KPI 3

### Test Population

**32,598**

Secondary:

> 1,380 new failures\
> 40 held-out wafers

------------------------------------------------------------------------

## KPI 4

### Resolution

**511 → 531 features**

Secondary:

> +20 sub-die signals

------------------------------------------------------------------------

# 17. Model Comparison

Use a clean visual progression.

``` text
MODEL A
Die + Spatial Baseline
511 features

500 Die Parametric
+ 10 Spatial
+ 1 Die Anomaly

            ↓

       +20 SUB-DIE SIGNALS

            ↓

MODEL B
Multi-Resolution Model
531 features

Model A
+ 19 Block Statistics
+ 1 Block Anomaly
```

The "+20" should use the Model B purple accent.

------------------------------------------------------------------------

# 18. Wafer Visualization

Keep the existing 4-panel visualization.

Order:

1.  Pre-test state
2.  Ground truth
3.  Model risk
4.  Spatial risk field

Add a compact explanatory legend.

### Labels

**PRE-TEST**\
What was known before burn-in

**GROUND TRUTH**\
What actually failed

**MODEL RISK**\
What the model predicts

**SPATIAL FIELD**\
Where risk clusters

Use consistent legends and axis labels.

------------------------------------------------------------------------

# 19. Hero Demo --- Why Model B Changed Its Mind?

This should be the most visually memorable interactive section.

Title:

> **Why Did Model B Change Its Mind?**

Subtitle:

> Sub-die evidence reveals localized variation that die-level averages
> can hide.

Use the benchmark:

### W_N_0066 · Die (2,13)

Three-column comparison:

``` text
MODEL A
12.9%
PASS

→

MODEL B
62.9%
FLAGGED

→

GROUND TRUTH
FAIL
```

Below:

> Model A sees the die-level average. Model B sees localized sub-die
> variation.

------------------------------------------------------------------------

# 20. Raw vs Calibrated Scores

Never mix these concepts.

Display:

``` text
RAW MODEL SCORE
12.9%

CALIBRATED RISK
4.5%
```

and:

``` text
RAW MODEL SCORE
62.9%

CALIBRATED RISK
45.2%
```

Use distinct labels.

Never call raw scores calibrated probabilities.

------------------------------------------------------------------------

# 21. Explainability

Primary question:

> **Why was this die flagged?**

First show the top 3 drivers.

Example:

``` text
blk_mean
↑ increases risk

sp_dist_to_fail
↓ reduces risk

blk_q75
↑ increases risk
```

Then show the detailed SHAP plot.

Spatial features must be labeled:

> Context --- not an intervention

------------------------------------------------------------------------

# 22. Domain Attribution

Use a clean four-row table:

  Domain                 Role
  ---------------------- ---------------------------
  Die Parametric         Primary signal
  Spatial Neighborhood   Context
  Sub-Die Block          Micro-structural evidence
  Anomaly Scores         Supporting signal

Do not make this look like a generic ML feature-importance table.

------------------------------------------------------------------------

# 23. Risk Sensitivity

Rename:

> Model-Based Counterfactual Sensitivity Analysis

to:

> **Risk Sensitivity --- What Drives the Score?**

Use a short explanation:

> If measurable signals move toward their normal range, how does the
> model score respond?

Keep the disclaimer:

> Model-based mathematical sensitivity estimate; not a physical
> manufacturing simulation or causal intervention.

------------------------------------------------------------------------

# 24. Sub-Die Signal Profile

Heading:

> **Sub-Die Evidence --- 2,000 Internal Readings**

Hero statistic:

**362 anomalous readings**

Secondary:

**18.1% of readings**

Show:

> Robust threshold: median ± 2×MAD

Keep the sequential-index disclaimer.

------------------------------------------------------------------------

# 25. Feature Audit Disclosure

Use a compact information card:

> **Feature audit**\
> Four block statistics were non-contributory in this experiment. They
> remain in the model definition for reproducibility.

This demonstrates scientific honesty.

------------------------------------------------------------------------

# 26. Triage

Heading:

> **Same Inspection Budget. More Failures Caught.**

At 10%:

### Model A

**768 / 1,380**

55.7%

### Model B

**834 / 1,380**

60.4%

### Hero delta

**+66 failures**

Secondary:

> 3,260 dies screened by each model.

The gains curve should visually reinforce the advantage.

------------------------------------------------------------------------

# 27. Triage Curve

Model A:

`--sd-model-a`

Model B:

`--sd-model-b`

Random baseline:

`--sd-text-muted`

At the selected budget, add a vertical guide using `--sd-red`.

Do not use red for an entire model curve.

------------------------------------------------------------------------

# 28. Status Components

Use compact badges.

### Verified

``` text
● VERIFIED
```

Green.

### Model B

``` text
MODEL B
```

Purple.

### Context

``` text
CONTEXT
```

Green/cyan.

### Failure

``` text
FAIL
```

Failure red.

### Calibrated

``` text
CALIBRATED
```

Neutral or cyan.

------------------------------------------------------------------------

# 29. Buttons

Primary:

-   SanDisk red background;
-   white text;
-   6--8px radius.

Secondary:

-   transparent;
-   1px border;
-   white/secondary text.

Hover:

-   red becomes slightly brighter;
-   translateY(-1px);
-   transition 120--180ms.

Do not use large pill buttons except for compact status controls.

------------------------------------------------------------------------

# 30. Inputs

Select boxes and radio controls should use:

-   dark surface;
-   1px border;
-   white text;
-   subtle red focus ring;
-   6--8px radius.

Focus state:

``` text
border: #F51B0B
box-shadow: 0 0 0 2px rgba(245,27,11,0.18)
```

------------------------------------------------------------------------

# 31. Micro-Animations

Animations should be subtle and purposeful.

## Page load

KPI numbers:

-   fade + translateY(6px);
-   duration 250--350ms.

## Card hover

-   translateY(-1px);
-   border becomes slightly brighter;
-   150ms.

## Model switch

When switching A ↔ B:

-   crossfade charts;
-   180--250ms.

## Wafer selection

Selected die:

-   subtle pulse around the die;
-   1--2 cycles only;
-   never continuous flashing.

## Triage slider

-   chart transition 200--300ms.

## Section entrance

Optional:

-   fade-in;
-   translateY(8px);
-   250ms.

Avoid:

-   spinning loaders for static content;
-   bouncing cards;
-   constant animations;
-   aggressive parallax;
-   neon glows.

------------------------------------------------------------------------

# 32. Motion Accessibility

Respect:

``` text
prefers-reduced-motion
```

If reduced motion is enabled:

-   remove transforms;
-   remove pulsing;
-   use instant transitions.

------------------------------------------------------------------------

# 33. Charts

Charts should follow one consistent visual language.

### Background

`--sd-surface-2`

### Grid

Low-contrast `--sd-border`

### Model A

Cyan

### Model B

Purple

### Ground truth

Failure red

### Positive context

Green

Avoid excessive chart decoration.

------------------------------------------------------------------------

# 34. Tables

Tables should look like engineering data tables.

Header:

-   `--sd-surface-3`
-   11--12px uppercase labels.

Rows:

-   `--sd-surface-2`
-   1px bottom border.

Hover:

-   slightly brighter surface.

Avoid zebra stripes unless necessary.

------------------------------------------------------------------------

# 35. Cards

Cards should be:

-   dark;
-   lightly bordered;
-   8--12px radius;
-   20--28px padding;
-   no excessive shadows.

Use one highlighted card for the most important result.

The +66 card should be visually dominant.

------------------------------------------------------------------------

# 36. Section Dividers

Use subtle horizontal rules.

Preferred:

``` text
────────────────────────────────────
```

Optionally place a tiny red pixel block at the left:

``` text
■ ──────────────────────────────────
```

This connects the UI to the pixel-driven brand language.

------------------------------------------------------------------------

# 37. Empty States

If data is unavailable:

``` text
NO DIE SELECTED

Select a die to inspect model risk,
feature attribution and sub-die evidence.
```

No generic Streamlit-style empty-state copy.

------------------------------------------------------------------------

# 38. Loading States

Use a small red/cyan linear progress indicator.

Avoid large spinners.

Example:

``` text
Loading wafer intelligence
━━━━━━━━━━━━━━░░░░
```

------------------------------------------------------------------------

# 39. Error States

Use:

-   subtle red left border;
-   clear title;
-   concise explanation;
-   recovery action.

Example:

> **Unable to load wafer profile**\
> The selected wafer data could not be loaded.

Do not expose Python stack traces to judges.

------------------------------------------------------------------------

# 40. Responsive Behavior

Desktop is the primary judging environment.

Still support:

### 1440px+

Full dashboard.

### 1024--1439px

Reduce sidebar width and card spacing.

### \<1024px

Stack KPI cards and collapse secondary panels.

Charts should remain readable.

------------------------------------------------------------------------

# 41. Accessibility

Minimum requirements:

-   strong contrast;
-   keyboard-focus states;
-   readable 14px minimum body text;
-   semantic headings;
-   color should not be the only indication of failure;
-   labels for charts;
-   tooltips for technical terms.

For example:

Do not show only:

`●`

Show:

`● FAIL`

------------------------------------------------------------------------

# 42. Product Copy

Use concise, engineering-oriented language.

Prefer:

> Same inspection budget. More failures caught.

Instead of:

> Model B consistently outperforms Model A across operational inspection
> budgets.

Prefer:

> Sub-die evidence reveals localized variation.

Instead of:

> Micro-structural block telemetry completely invisible to die-level
> parametric sensors.

Prefer:

> Raw score

instead of:

> Probability

when referring to uncalibrated LightGBM outputs.

------------------------------------------------------------------------

# 43. Scientific Honesty

Never claim:

-   causal intervention;
-   physical process control;
-   production-ready fab deployment;
-   a theoretical F1 ceiling unless formally demonstrated;
-   monetary savings unless measured;
-   raw score = probability;
-   every feature contributes;
-   sub-die index = physical 3D position.

The design system should visually reinforce transparency.

------------------------------------------------------------------------

# 44. Footer

Do NOT use:

> SanDisk Die Yield Prediction Hackathon Deliverable

Preferred:

No footer.

If a footer is technically required:

``` text
DieYield Intelligence
Multi-Resolution Yield Prediction
Model evaluation • v1.0
```

Keep it extremely small and low contrast.

------------------------------------------------------------------------

# 45. Judge Presentation Flow

The dashboard should naturally guide a judge through this order:

## 01

**What is the result?**

Model B: +0.0337 PR-AUC

## 02

**What changed?**

+20 sub-die signals

## 03

**Does it matter operationally?**

+66 failures at the same 10% inspection budget

## 04

**Can I understand the prediction?**

SHAP + spatial context

## 05

**Why is Model B better?**

W_N_0066 B-only catch

## 06

**Can I inspect the evidence?**

2,000-reading sub-die profile

## 07

**Are you honest about limitations?**

Semi-synthetic benchmark + feature audit + prototype disclosure

------------------------------------------------------------------------

# 46. Final Visual Principle

Every screen should answer:

> **What should the judge notice first?**

Use this hierarchy:

### 1. Outcome

Big number.

### 2. Explanation

Short sentence.

### 3. Evidence

Chart/map/table.

### 4. Detail

Technical metadata.

The interface should never make a judge read a paragraph before
understanding the result.

------------------------------------------------------------------------

# 47. Design Token Summary

``` css
:root {
  --sd-red: #F51B0B;
  --sd-black: #050505;
  --sd-surface-1: #0D0F12;
  --sd-surface-2: #15181D;
  --sd-surface-3: #1C2128;
  --sd-border: #30353D;

  --sd-white: #FFFFFF;
  --sd-text-secondary: #B7BDC7;
  --sd-text-muted: #7F8792;

  --sd-success: #35D07F;
  --sd-warning: #F4B740;
  --sd-failure: #FF5A5F;

  --sd-model-a: #25B9E6;
  --sd-model-b: #A855F7;
  --sd-context: #43D17C;

  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 12px;

  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --space-8: 64px;
}
```

**Note:** The hex values above are the project's implementation tokens
derived from the supplied screenshots and the intended visual direction.
They should be treated as the project's design-system values, not as a
claim that SanDisk has publicly published these exact UI hex codes.

------------------------------------------------------------------------

# 48. Reference Basis

The design direction is based on:

1.  Supplied screenshots of the current SanDisk website, including the
    red navigation/header, black hero/footer, high-contrast white
    typography, and restrained product-page composition.
2.  SanDisk's public 2024/2025 brand announcement describing its new
    "Mindset of Motion," pixel-driven mark, clean lines, open
    letterforms, angular geometry, and future-facing visual language.
3.  SanDisk's public "Behind the Design" explanation describing the
    pixel as a core design concept, broad geometric/slab-serif forms,
    and a warmer red intended to feel immediate and energetic.
4.  SanDisk's 2026 public design communications emphasizing purposeful,
    human-centered product design and performance-oriented technology
    experiences.

The design should borrow **principles**, not copy the corporate website.

------------------------------------------------------------------------

# 49. Definition of Done

The UI is considered design-complete when:

-   [ ] No hackathon/internal-deliverable footer appears.
-   [ ] Product identity is visible.
-   [ ] SanDisk-inspired red/black/white system is consistent.
-   [ ] Typography is consistent.
-   [ ] Model A and Model B colors are consistent.
-   [ ] +66 is the dominant operational result.
-   [ ] 768 vs 834 is consistent everywhere.
-   [ ] Raw and calibrated scores are clearly separated.
-   [ ] W_N_0066 is visually strong.
-   [ ] Sidebar looks like a product control center.
-   [ ] Emojis are reduced.
-   [ ] Cards share consistent radius, spacing and borders.
-   [ ] Charts share a consistent color system.
-   [ ] Motion is subtle and purposeful.
-   [ ] Reduced-motion behavior is respected.
-   [ ] No unsupported scientific claims are introduced.
-   [ ] Existing ML artifacts and calculations remain untouched.
-   [ ] The application feels like an industrial semiconductor
    intelligence product rather than a student dashboard.
