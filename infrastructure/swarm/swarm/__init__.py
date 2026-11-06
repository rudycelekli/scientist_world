"""Swarm — coordinates the builder agents that turn a paper into an environment.

For one paper, Swarm drives a workflow of builder roles (reproducer,
designer, packager, verifier) and, within each role, the loop: invoke
worker → read action.yaml → submit job via Nexus → wait → invoke worker
again. Across papers, it runs many such builds in parallel from a queue.
"""

__version__ = "0.1.0"
