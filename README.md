# ML From Scratch — Intuition First

Most ML resources are either math-heavy textbooks or shallow `model.fit()` tutorials. This course is the missing middle: **plain-language intuition first, then the math with every symbol pre-explained, then a from-scratch implementation whose code comments point back to the intuition, then how to actually read the results.**

## The format (identical in every folder)

Each model gets one folder containing:

```
NN-model-name/
├── README.md                      "In one sentence" + how-to-read guide (time-budgeted paths)
│                                  §0 glossary — every term in plain words BEFORE any formula
│                                  §1 what the model ASSUMES + real-world examples
│                                  §2 the intuition, in pictures   ⏸ checkpoint → run code
│                                  §3 the math, motivated          ⏸ checkpoint → run code
│                                  §4 every knob/parameter + what breaking it looks like
│                                  §5 reading the results (what each metric actually says)
│                                  §6 when NOT to use it + how to notice
│                                  §7 README → notebook map
│                                  §8 interview prep (30-second answer + Q&A)
│                                  §9 common mistakes (the learner's, not the model's)
├── assets/                        the visual explanations
├── <model>_from_scratch.ipynb     NumPy-only build; every block cites its README §
├── <model>_with_library.ipynb     same model in a few library lines, verifying the scratch build
└── data/                          a small real-feeling dataset to run everything on
```

**Rules of the course:**
1. No symbol appears in a formula before it has appeared in plain language.
2. Every formula gets a "read aloud" translation.
3. Every notebook block cites the README section it implements.
4. Every model gets broken on purpose at least once (bad learning rate, wrong assumption, etc.) — failures teach more than successes.
5. The library version must reproduce the scratch version — proving there is no magic.
6. Never demand one continuous 30-minute read: every lesson opens with time-budgeted reading paths and has ⏸ checkpoints that alternate reading with running code.
7. Every lesson ends interview-ready: a one-sentence summary, a 30-second spoken answer, Q&A, and the learner's common mistakes.

## Roadmap

| # | Model | Status | The ONE new idea it introduces |
|---|---|---|---|
| 01 | Linear Regression | ✅ | The learning loop itself: predict → error → gradient → update |
| 02 | Logistic Regression | ✅ | Squashing a line into a probability (sigmoid) |
| 03 | K-Nearest Neighbors | ✅ | Learning without a loop: memory + similarity |
| 04 | Naive Bayes | ✅ | Prediction as counting evidence (probability flips) |
| 05 | Decision Trees | ✅ | Learning as asking the best questions (entropy/gini) |
| 06 | Random Forest | ✅ | Wisdom of crowds: many weak models vote |
| 07 | SVM | ✅ | The widest street between classes (margins, kernels) |
| 08 | Gradient Boosting | ✅ | Each model fixes the previous one's mistakes |
| 09 | K-Means & PCA | ✅ | Learning without labels |
| 10 | Neural Network | ✅ | Stacking lesson 01 with bends; backprop = chain rule |
| 11 | CNNs | ✅ | Reusing weights across space (seeing) |
| 12 | RNN / LSTM | ✅ | Reusing weights across time (remembering) |
| 13 | Attention | ✅ | Letting inputs look at each other |
| 14 | Transformers | ✅ | Attention is all you need — the modern engine |
| 15 | Vision Transformers | ✅ | Images as sentences of patches |
| 16 | Multimodal (CLIP-style) | ✅ | One space for images AND text |
| 17 | Modern LLM concepts | ⬜ | Pretraining, fine-tuning, RLHF — how chatbots are made |

Each lesson assumes only the lessons before it. The deep-learning half (10+) leans hard on lesson 01: it is the same predict → error → gradient → update loop, scaled up.
