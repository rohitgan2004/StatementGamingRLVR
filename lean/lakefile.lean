import Lake
open Lake DSL

package «leandrift» where
  -- Pinned for all runs; see README for the exact Mathlib/REPL revisions.
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

-- Mathlib provides the theorem library the corpus is stated against.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.15.0"

-- The REPL is the JSON server the warm worker pool (leandrift/lean/repl.py) drives.
require repl from git
  "https://github.com/leanprover-community/repl.git" @ "master"

@[default_target]
lean_lib «LeanDrift» where
  -- Pinned import surface used by every episode (import policy is harness-fixed).
  roots := #[`LeanDrift]
