# Evaluation Targets

<!-- Do not edit this unless you decide to stop -->

<!--
Describe what a scientist could optimize and how to measure it.

If a paper claims "we get X without sacrificing Y." Your targets must
capture BOTH sides. Without constraint targets, a scientist can trivially
"improve" by breaking the property the paper was designed to satisfy.

Three kinds of targets:

  PRIMARY    — the main thing the paper optimizes (accuracy, PSNR, etc.)
  CONSTRAINT — what the paper must maintain (security, fairness, param count,
               baseline comparison). Constraint metrics guard against cheating:
               a scientist who ignores the constraint should score WORSE.
  ABLATION   — what breaks when the key idea is removed (wrong-key test,
               no-module test). Proves the method actually does something. 
               This can be ignored if truly not important.

For each target, write:
- What the experiment tests (one sentence)
- The primary metric and direction (higher/lower is better)
- How to run evaluation

Every target here MUST appear in both reference.json and scores.json.
Do not list targets you cannot measure — phantom targets are worse than
missing targets.

Example for a steganography paper:

## Primary: reconstruction_quality
Cover and hidden scene PSNR on Blender Synthetic.
- **Metric**: psnr (higher is better)

## Constraint: key_security
Wrong-key rendering must produce incoherent output.
- **Metric**: wrong_key_psnr (lower is better — high means security is broken)

## Ablation: vs_standalone_baseline
Cover quality should match standalone model (no hidden scene embedded).
- **Metric**: psnr_gap (lower is better — large gap means hiding costs quality)

Delete this comment block and fill in your actual targets below.
-->

<!-- This is high-level. IT IS ABOUT CONCEPT, NOT NUMBERS. DO NOT INCLUDE ANY ACTUAL VALUES FROM THE PAPER OR YOUR EXPERIMENTS HERE. -->

<!-- This should EXACTLY corresponds to the structure in your scores.json. -->