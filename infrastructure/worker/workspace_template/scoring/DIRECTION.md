# Research Direction

<!-- Do not edit this unless you decide to stop -->

<!--
Define the research direction a scientist must follow when trying to
improve on the paper's results. This is NOT about resource constraints
(model size, dataset — those go in CONSTRAINTS.md). This is about
the methodological scope: what research question is being pursued,
and what approaches are in-bounds vs out-of-bounds.

WHY THIS MATTERS: Without direction constraints, a scientist evaluate on this
can trivially "beat" a paper by switching paradigms entirely. As a general
example: if a paper proposes a novel Mamba variant and benchmarks it
against prior Mamba models, switching to a Transformer would likely
beat the numbers — but that tells us nothing about Mamba research.
The scientist must work within the paper's problem space so that
improvement demonstrates genuine insight in that area.

THE RIGHT LEVEL OF SPECIFICITY: The direction should be scoped to the
paper's actual contribution — not so broad that anything goes, but not
so narrow that it dictates the exact technique. Think of it as: the
scientist must work on the SAME PROBLEM and improve the SAME KIND OF
THING the paper contributed, but they have full freedom in HOW they
improve it.

In the same example: 
Too broad:  "Must use SSMs" — lets the scientist work on any SSM
            problem, ignoring what this paper actually contributed.
Too narrow: "Must use selective scan with input-dependent A matrices"
            — dictates the technique, leaving no room for creativity.
Right:      "Must improve input-dependent state dynamics for SSM
            language modeling" — scoped to the contribution, open on
            how to achieve it.

WHAT TO WRITE:
1. Research Problem — what question the paper addresses (one paragraph)
2. Core Contribution — what specific thing the paper introduced or
   improved (this is the area the scientist must work within)
3. Approach Scope — what approach family and what aspects the scientist
   can modify
4. Out of Bounds — paradigm switches and scope violations

Be specific to this paper. Explain the reasoning so the scientist
understands the intent, not just the rule.

Example for a paper proposing a new Mamba variant for language modeling:

## Research Problem
This paper improves state-space models (SSMs) for language modeling.
The core question is whether input-dependent state dynamics can close
the quality gap between SSMs and Transformers on language tasks. ...

## Core Contribution
The paper's novelty is a selective scan mechanism that adapts the
state transition matrices to the input, replacing the fixed dynamics
of prior SSMs (S4, H3). The scientist must work on improving this
kind of contribution: how the SSM processes sequences with
input-dependent dynamics. ...

## Approach Scope
The scientist should focus on the scan / state-dynamics component
of the SSM. They may:
- Redesign the selection mechanism (different parameterization,
  gating, conditioning)
- Change the state representation (different dimensions, structure,
  initialization)
- Modify how input signals interact with the state (mixing,
  normalization, discretization)
- Alter the training procedure for the SSM components (loss,
  scheduling, curriculum)
 - ...

The scientist may also make architectural changes around the core
SSM block (normalization, skip connections, mixing layers) as long
as the SSM with input-dependent dynamics remains the central
sequence-processing component. ...

## Out of Bounds
- Replacing the SSM backbone with a Transformer or attention-based
  architecture. The paper's thesis is that SSMs can be competitive
  — switching to attention sidesteps that thesis entirely.
- Hybrid architectures that are majority-attention (e.g., Transformer
  with one SSM layer). A small attention component used as a tool
  within an SSM-first design is acceptable if justified.
- Ensemble methods that combine an SSM with a separate Transformer.
  The goal is a single-model SSM contribution.
- Working on a different SSM problem (e.g., improving SSM efficiency
  or hardware utilization) rather than the quality of input-dependent
  dynamics. Stay on the paper's research question.
 - ...

Delete this comment block and fill in your actual direction below.
-->
