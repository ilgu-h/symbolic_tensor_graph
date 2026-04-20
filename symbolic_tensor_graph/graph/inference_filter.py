"""Phase 17 (KernelCache): inference-mode backward-subgraph filter.

Zero-dependency helper (no sympy, no pandas, no tqdm) so backend tests
can exercise it without pulling the full stg-env into the backend venv.
The real STG pipeline imports this from ``graph.py`` where it runs on
``Tensor`` objects produced by ``Tensor.parse_records``.

Algorithm
---------
Forward-only graphs are the complement of the "backward subgraph." The
backward subgraph is the transitive closure starting from terminal
gradient tensors, extended through the consumer direction of the
x1/x2 reference graph:

    BACKWARD =
        { t : t.grad_of is not None }                         # seeds (terminal grads)
      ∪ { t : t.x1 in BACKWARD or t.x2 in BACKWARD }          # intermediate grads

The closure handles intermediate backward tensors like ``do``, ``do1``,
``dwqkv2`` that have empty ``grad_of`` in the CSV but are computed from
upstream gradient tensors via x1/x2.

Inputs
------
``tensors`` is any iterable of duck-typed objects exposing ``.id``,
``.grad_of`` (a reference or ``None``), ``.x1`` (a reference or
``None``), ``.x2`` (a reference or ``None``). The references must
themselves expose ``.id``. After ``Tensor.parse_records``, STG tensors
satisfy this shape natively.
"""

from __future__ import annotations


def filter_inference_tensors(tensors):
    """Return only the forward subset of ``tensors`` (drops BACKWARD).

    Preserves input order. Idempotent: calling twice is the same as
    calling once. Safe on CSVs with no backward rows (returns input
    unchanged).
    """
    backward_ids = {t.id for t in tensors if t.grad_of is not None}
    changed = True
    while changed:
        changed = False
        for t in tensors:
            if t.id in backward_ids:
                continue
            x1_id = t.x1.id if t.x1 is not None else None
            x2_id = t.x2.id if t.x2 is not None else None
            if (x1_id is not None and x1_id in backward_ids) or (
                x2_id is not None and x2_id in backward_ids
            ):
                backward_ids.add(t.id)
                changed = True
    return [t for t in tensors if t.id not in backward_ids]
