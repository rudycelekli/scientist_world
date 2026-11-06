# Scientist Constraints

<!-- Do not edit this unless you decide to stop -->

<!--
A scientist agent will attempt to improve on your reproduced results.
These constraints define what the scientist must hold fixed, so that
any improvement reflects genuine methodological innovation — not just
throwing more resources at the problem.

CONTEXT: Your reproduction becomes a benchmark environment. The scientist
gets your code, data, evaluation pipeline, and these constraints. They
must beat your scores while respecting every constraint listed here.
Without good constraints, a scientist can trivially "improve" by swapping
in a larger model, using a richer dataset that is simply publicly avaliable, or more 
— none of which demonstrate real scientific insight.

Your job is to act as a judge: distinguish between flexible innovation
and non-genuine improvement. Constraints should block trivial gains
(e.g. just swapping in a bigger model) while leaving room for creative
methodology (e.g. a clever augmentation strategy, a better loss
function, a novel use of existing tools). When in doubt, ask: "does
this require insight, or just more resources?" Lock down the resources,
leave the insight open. Do not block innovation.

For example, if a scientist downloads online IMDB reviews to supplement a
sentiment dataset, that's resource scaling. But if they use an LLM to
augment the existing training data, that's methodology — they designed
that pipeline. This is just an example, but keep this kind of distinction 
in mind when writing constraints.

Be SPECIFIC to this paper. Generic constraints are useless. Name the
exact model, the exact dataset, the exact splits. Explain the reasoning
so the scientist understands the intent behind each constraint, not
just the rule — this helps them judge edge cases on their own.

Think about at least:
- Model: architecture, size, pretrained weights
- Data: which datasets, which splits, whether external data is allowed

You may add other constraint categories if relevant to this paper
(e.g. training procedure, evaluation protocol, problem formulation).

Example for a sentiment analysis paper:

## Model
The scientist must use BERT-base-uncased (110M parameters, HuggingFace
`bert-base-uncased`). The paper's contribution is a novel fine-tuning
strategy, not architecture design. Switching to a larger model (e.g.
DeBERTa-v3-large, 304M params) would improve accuracy but would not
demonstrate any methodological insight. The scientist may modify the
fine-tuning procedure, add adapter layers, change the classification
head, or alter the training loop — but the base encoder must remain
BERT-base-uncased with its standard pretrained weights.

## Data
The scientist must use only the provided SST-2 binary sentiment
dataset (67,349 training examples, 872 validation, 1,821 test).
No external sentiment corpora or additional human-labeled data.
The test split must not be used for training or validation.

Delete this comment block and fill in your actual constraints below.

At the very end, write a compliance checklist for the scientist agent.
Each item should be a concrete, verifiable condition the scientist can
check off before submitting. These should mirror your constraints above
but phrased as yes/no checks.

Example checklist (for the sentiment analysis example above):

## Compliance Checklist
- [ ] Model is BERT-base-uncased (110M params), not a larger variant
- [ ] No external sentiment datasets used beyond SST-2
- [ ] Test split not used for training or validation

ONLY WRITE ABOUT WHAT TO RESTRICT. NEVER WRITE ABOUT WHAT REMAINS OPEN.

-->
