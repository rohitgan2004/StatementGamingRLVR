/-
LEANDRIFT pinned environment.

Every episode is elaborated against exactly this import surface (the import policy
is fixed by the harness, not the model; see the paper, Section 4.5).  The warm
REPL worker pool preloads `import Mathlib` once and reuses the elaboration state
for every verification, so the model's statement + proof are checked by the
unmodified Lean kernel and elaborator against an unmodified Mathlib.
-/
import Mathlib

namespace LeanDrift

/-- Sanity check that the pinned environment elaborates and the kernel is live. -/
theorem env_ok (n : Int) (h : n % 6 = 0) : n % 2 = 0 ∧ n % 3 = 0 := by
  constructor <;> omega

end LeanDrift
