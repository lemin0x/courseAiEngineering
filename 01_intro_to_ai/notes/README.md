# Introduction to AI · Notes


## 1. WHAT IS ARTIFICIAL INTELLIGENCE?

### Formal Definitions

| Source | Definition |
|--------|------------|
| **John McCarthy (1956)** | "The science and engineering of making intelligent machines." |
| **Stuart Russell & Peter Norvig** | AI is the study of agents that receive percepts from the environment and perform actions. |
| **Modern working definition** | AI is a system's ability to interpret external data, learn from that data, and use those learnings to achieve specific goals through flexible adaptation. |

### The Four Approaches to AI (Russell & Norvig's framework)

This is a fundamental way to understand what "intelligence" means in AI:

| | Human-centered | Rational-centered |
|--|---------------|-------------------|
| **Thinking** | Systems that think like humans (cognitive modeling) | Systems that think rationally (formal logic) |
| **Acting** | Systems that act like humans (Turing Test approach) | Systems that act rationally (rational agent approach) |

**Key insight:** Modern AI (especially ML/DL) mostly follows the **rational agent approach** — we don't require the system to think the way humans do, only to act optimally to achieve goals.

---

## 2. TYPES OF AI — THE THREE LEVELS

### ANI: Artificial Narrow Intelligence
- **What it is:** AI specialized in a single, narrow domain.
- **Where we are:** Everything you see today is ANI.
- **Examples:**
  - AlphaGo (world champion at Go, cannot play chess without retraining)
  - ChatGPT (powerful language model, cannot drive a car)
  - Tesla Autopilot (drives cars, cannot write poetry)
  - Google Translate, facial recognition, recommendation algorithms
- **Key feature:** No consciousness, no true understanding, no ability to generalize outside its training domain.

### AGI: Artificial General Intelligence
- **What it is:** A system with human-level cognitive abilities across any task a human can do.
- **Current status:** Does not exist anywhere.
- **What it requires (that we don't have yet):**
  - Transfer learning across completely different domains
  - Common sense reasoning
  - Causal understanding (not just correlation)
  - Continuous learning without catastrophic forgetting
- **Debate:** Some argue LLMs show "sparks of AGI" (Microsoft Research, 2023), but this is highly contested. Most experts agree we are not close.

### ASI: Artificial Super Intelligence
- **What it is:** Intelligence surpassing the brightest human minds in every field.
- **Characteristics (theoretical):**
  - Recursive self-improvement
  - Orders of magnitude faster thinking
  - Access to all human knowledge instantly
- **Timeline debate:** Predictions range from 2040 to never.
- **Importance:** This is why AI safety/alignment research exists — better to solve alignment before we build ASI than after.

---

## 3. AI vs MACHINE LEARNING vs DEEP LEARNING

This taxonomy is essential to understand:

- **Artificial Intelligence (AI):** Any technique that enables computers to mimic aspects of human intelligence. This includes rule-based systems, search algorithms, logic, planning, and machine learning.
  - **Machine Learning (ML):** Methods that learn patterns from data instead of relying on hand-coded rules. Examples: linear regression, decision trees, SVMs, random forests, k-means.
    - **Deep Learning (DL):** A subfield of ML that uses multi-layer neural networks to learn directly from raw, unstructured data (images, audio, text). Examples: CNNs, RNNs, Transformers, GANs, diffusion models.

Why this distinction matters
- **Rule-based AI (non-ML):** Programmers write explicit if–then rules. Works for well-defined problems but fails on perception tasks.
- **ML:** The machine infers the rules from examples; this data-driven approach enabled modern AI breakthroughs.
- **DL:** Excels on raw, high-dimensional data with minimal manual feature engineering and underlies many recent advances.



### Why this distinction matters
- **Rule-based AI (non-ML):** Programmers write explicit if-then rules. Works for simple, well-defined problems (chess engines in the 90s, tax software). Fails on perception tasks.
- **ML:** The machine figures out the rules from examples. This is what made modern AI possible.
- **DL:** Handles raw, unstructured data (images, audio, text) with minimal feature engineering. Responsible for the 2012–now revolution.

### Data Science positioning
Data Science overlaps with AI/ML but is distinct:
- **Data Science focus:** Extracting insights, making decisions, data pipelines, visualization.
- **AI/ML focus:** Building systems that perform tasks autonomously.
- They share: Statistics, Python, ML algorithms, data preprocessing.

---
## 4. THE FOUR PARADIGMS OF MACHINE LEARNING

Since Section 04 is dedicated to ML, here we focus on the conceptual understanding.

### 4.1 Supervised Learning
**Analogy:** A student learning with an answer key.

- Input data comes with **labels** (the correct answer).
- The model learns a mapping: **X → y**
- **Process:**
  1. Training: Show examples (features + label), model adjusts internal parameters
  2. Testing: Show new examples (features only), model predicts label
- **Two main types:**
  - **Regression:** Predict continuous values (e.g., house price: $450,000)
  - **Classification:** Predict categories (e.g., email is spam or not spam)
- **Examples:** Medical diagnosis from symptoms, image recognition with labeled images, fraud detection.

### 4.2 Unsupervised Learning
**Analogy:** A student grouping books by topic without labels.

- Input data has **no labels.**
- The model finds hidden structure/patterns.
- **Main techniques:**
  - **Clustering:** Group similar data points (K-Means, DBSCAN)
  - **Dimensionality Reduction:** Compress data while preserving structure (PCA, t-SNE)
  - **Anomaly Detection:** Find unusual data points
- **Examples:** Customer segmentation, recommendation systems (people who bought X also bought Y), genetic clustering.

### 4.3 Semi-Supervised Learning
**Analogy:** A student who has an answer key for only a few questions, but must figure out the rest from patterns.

- Combines a **small amount of labeled data** with a **large amount of unlabeled data.**
- **Why it matters:** Labeling data is expensive and time-consuming. Semi-supervised learning provides a practical middle ground.
- **How it works:**
  1. Train initial model on the small labeled dataset
  2. Use the model to generate **pseudo-labels** for the unlabeled data
  3. Retrain on the expanded dataset
  4. Repeat until convergence
- **Key assumption:** Data points close to each other likely share the same label (smoothness/clustering assumptions).
- **Examples:**
  - Medical image analysis (few expert-labeled scans, thousands of unlabeled ones)
  - Web page classification (a few human-categorized pages, millions uncategorized)
  - Speech recognition with limited transcribed audio
- **Modern relevance:** A core idea behind how LLMs are sometimes fine-tuned and how self-supervised pre-training (like BERT, GPT) creates representations later used for supervised tasks.

### 4.4 Reinforcement Learning
**Analogy:** Training a dog with treats.

- An **agent** interacts with an **environment.**
- It takes **actions** and receives **rewards** or **penalties.**
- Goal: Learn a **policy** that maximizes cumulative reward over time.
- **Key concepts:**
  - Exploration vs Exploitation tradeoff
  - Delayed rewards (action now, reward later)
  - State, Action, Reward, Next State (SARSA)
- **Famous examples:** AlphaGo, ChatGPT (RLHF — Reinforcement Learning from Human Feedback), robot locomotion, game-playing AI.

### Comparison Table

| Feature | Supervised | Unsupervised | Semi-Supervised | Reinforcement |
|---------|-----------|--------------|-----------------|----------------|
| Data | Labeled | Unlabeled | Small labeled + large unlabeled | Environment feedback |
| Goal | Predict output | Discover patterns | Predict output (with less labeling cost) | Maximize reward |
| Feedback | Immediate, correct answer | None | Partial (from labeled portion) | Delayed, scalar reward |
| Human effort | High (labeling) | Low | Medium (some labeling) | Medium (design reward function) |
| Key challenge | Need large labeled datasets | Hard to evaluate results | Getting pseudo-labels right | Exploration vs exploitation |



---

## 5. A CONCISE HISTORY OF AI

This is not just trivia — it explains *why* the field looks the way it does today.

### The Dawn (1950–1956)
- **1950:** Alan Turing publishes "Computing Machinery and Intelligence," proposes the Turing Test.
- **1956:** Dartmouth Summer Workshop — John McCarthy coins "Artificial Intelligence." Attendees include Marvin Minsky, Claude Shannon. This is considered AI's birth.

### Golden Age (1956–1974)
- Early optimism: "Machines will be capable, within twenty years, of doing any work a man can do" (Herbert Simon, 1965).
- ELIZA (1966): Early chatbot simulating a psychotherapist.
- Shakey the Robot: First general-purpose mobile robot that could reason about its actions.

### First AI Winter (1974–1980)
- **Causes:**
  - Computers too weak to deliver on promises.
  - Combinatorial explosion in search problems.
  - Lighthill Report (UK) heavily criticized AI progress — funding slashed.
- **Lesson:** Overpromising kills progress.

### Expert Systems Boom (1980–1987)
- Rule-based systems encoding human expert knowledge.
- Successes: R1/XCON (saved DEC $40M/year configuring computer orders).
- **Problem:** Brittle, hard to maintain, couldn't handle uncertainty well.

### Second AI Winter (1987–1993)
- Expert systems collapsed when desktop computers became powerful enough to replace specialized Lisp machines.
- DARPA cut AI funding.

### The Rise of Machine Learning (1990s–2010)
- Shift away from logic-based AI to data-driven approaches.
- **1997:** IBM Deep Blue defeats Kasparov at chess (search + specialized hardware).
- Statistical NLP becomes dominant over rule-based approaches.

### The Deep Learning Revolution (2012–2017)
- **2012:** AlexNet wins ImageNet by a huge margin — deep learning + GPUs prove decisive. This is the inflection point.
- Word2Vec, GANs, ResNet follow.
- **2014:** DeepMind acquired by Google.
- **2016:** AlphaGo defeats Lee Sedol.

### The Transformer Era (2017–Present)
- **2017:** "Attention Is All You Need" introduces the Transformer architecture.
- **2018:** BERT, GPT-1
- **2020:** GPT-3 shows emergent abilities at scale.
- **2022:** ChatGPT released, ushering in the generative AI era.
- **2023–2025:** Multimodal models (GPT-4, Gemini, Claude), open-source rivals (Llama, Mistral), AI agents, reasoning models.

---

## 6. MAJOR APPLICATION DOMAINS

Understanding these will help you contextualize every technique you learn later.

### Natural Language Processing (NLP) → Section 06
- Machine translation, chatbots, summarization, sentiment analysis
- Core tech: Transformers, LLMs, RAG

### Computer Vision → Section 07
- Image classification, object detection, segmentation, image generation
- Core tech: CNNs, ViTs, Diffusion Models

### Speech & Audio
- Speech-to-text, text-to-speech, music generation
- Core tech: RNNs (legacy), Transformers

### Recommendation Systems
- Product recommendations, content ranking
- Core tech: Collaborative filtering, two-tower neural networks

### Autonomous Systems
- Self-driving cars, drones, robotics
- Core tech: Sensor fusion, reinforcement learning, computer vision

### Generative AI → Sections 07 & 09
- Text generation, image/video generation, code generation
- Core tech: LLMs, Diffusion Models, GANs

---

## 7. ETHICS & AI SAFETY (Introduction)

This section will reappear throughout the course, but early awareness is critical.

### 7.1 Fairness and Bias
- Models learn biases present in training data.
- **Example:** A hiring model trained on historical data may discriminate against women if historical hiring was biased.
- Mitigation: Diverse datasets, bias audits, fairness metrics.

### 7.2 Transparency & Explainability
- Deep neural networks are often "black boxes."
- **Problem:** If an AI denies your loan, you deserve to know why.
- Fields: Explainable AI (XAI), interpretable ML.

### 7.3 Privacy
- Models can memorize and leak training data.
- Large models raise concerns about data usage.
- Approaches: Differential privacy, federated learning.

### 7.4 Misinformation and Deepfakes
- Generative AI makes it trivially easy to create convincing fake content.
- This is a major societal challenge being actively researched.

### 7.5 AI Safety and Alignment
- The **alignment problem:** How do we ensure AI systems do what we actually want, not just what we literally specify?
- **Paperclip maximizer** thought experiment: An ASI told to "maximize paperclip production" could theoretically destroy humanity to build more paperclips.
- This sounds like science fiction, but misalignment in narrow AI (e.g., social media optimizing for engagement at all costs, damaging mental health) is already here.

### 7.6 Job Displacement
- Historical precedent: Technology eliminates some jobs but creates others.
- AI may differ due to the speed and breadth of disruption (affecting white-collar cognitive work).
- Discussions center on retraining, UBI, and human-AI collaboration.

---


## 8. THE AI PROJECT WORKFLOW (Preview)

A concise, repeatable workflow you will use throughout the course:

1. Problem definition
  - Clarify whether AI is necessary, define the success metric, scope, and constraints.

2. Data collection & preparation
  - Gather, clean, label, and augment data. (Often ~80% of the effort.)

3. Model selection & training
  - Choose candidate architectures, train models, and tune hyperparameters.

4. Evaluation
  - Validate on held-out data, measure chosen metrics, and check for fairness and robustness.

5. Deployment (MLOps — Section 08)
  - Serve the model, add monitoring, logging, CI/CD, and plan for retraining.

6. Iteration
  - Continuously monitor production performance, collect new data, fix drift, and improve the system.

Quick checklist
- Define metric and acceptance criteria.
- Version data, code, and model artifacts.
- Add monitoring and alerting for data and model drift.
- Include privacy, safety, and reproducibility plans.


---

## 9. KEY VOCABULARY CHECKLIST

Before moving to Section 02, make sure you can explain these terms:

| Term | Can you define it? |
|------|-------------------|
| Artificial Intelligence | ☐ |
| ANI / AGI / ASI | ☐ |
| Machine Learning | ☐ |
| Deep Learning | ☐ |
| Supervised / Unsupervised / Reinforcement Learning | ☐ |
| Labeled vs Unlabeled Data | ☐ |
| Regression vs Classification | ☐ |
| Neural Network | ☐ |
| Feature | ☐ |
| Model | ☐ |
| Training / Testing | ☐ |
| Overfitting | ☐ |
| AI Ethics / Bias / Alignment | ☐ |
| Turing Test | ☐ |
| Transformer | ☐ |

---

## 10. RECOMMENDED SUPPLEMENTARY RESOURCES

- **Read:** Stuart Russell & Peter Norvig — *Artificial Intelligence: A Modern Approach* (Chapters 1–2)
- **Watch:** "AlphaGo — The Movie" (documentary, free on YouTube) — excellent motivation
- **Interactive:** Try Google's "Teachable Machine" to build a no-code image classifier in 5 minutes
- **Paper (approachable):** "On the Opportunities and Risks of Foundation Models" (Stanford CRFM, 2021)

