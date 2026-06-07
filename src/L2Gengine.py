# Removed all LLM related objects from this file to be able to run my experiments 

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
import torch
from torch import Tensor

# ====== DATA STRUCTURES (PYDANTIC) ============================================

class Preference(BaseModel):
    """Rappresenta una preferenza: preferred ≻ other."""
    preferred_candidate: str
    other_candidate: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ConstraintSpec(BaseModel):
    """Specifica del vincolo derivato dal linguaggio naturale."""
    id: str = Field(default_factory=lambda: "const_idx")
    subfunction_id: str
    constraint_type: Literal["upper_bound_absolute", "upper_bound_relative", "directional_improvement"]
    operator: Literal["<=", ">="] = "<="
    reference_candidate: Optional[str] = None
    threshold: Optional[float] = None
    margin: float = 0.0
    weight: float = 1.0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class L2GOutput(BaseModel):
    """Output strutturato atteso dall'oracle."""
    feedback_type: Literal["both", "preference_only", "direction_only", "none"]
    preference: Optional[Preference] = None
    constraints: List[ConstraintSpec] = Field(default_factory=list)

# ====== L2G ENGINE ============================================================

class L2GEngine:
    def __init__(
        self,
        context: Dict[str, Any],
        sub_surrogates: Dict[str, Callable[[Tensor], Tensor]],
        llm: Optional[Any] = None,  # kept for API compatibility, not used
        min_confidence: float = 0.5,
    ):
        self.context = context
        self.sub_surrogates = sub_surrogates
        self.min_confidence = min_confidence

        # State storage
        self.preference_history: List[Tuple[Tensor, Tensor]] = []  # (x_pref, x_other)
        self.constraint_specs: List[ConstraintSpec] = []

    # -------------------------------------------------------------------------
    # 1) PROCESS FEEDBACK
    # -------------------------------------------------------------------------

    def process_feedback(
        self,
        candidates: Dict[str, Tensor],
        feedback_text: str = "",
        simulated_output: Optional[Dict[str, Any]] = None,
    ):
        """
        Main entry point. Parses oracle output -> Updates internal state.
        Always pass simulated_output when using the oracle (LLM is disabled).
        """
        if simulated_output is None:
            raise ValueError(
                "LLM is disabled. You must pass simulated_output from the oracle."
            )

        parsed_output = L2GOutput(**simulated_output)

        # 2. Aggiorna Preferenze
        if parsed_output.feedback_type in ("both", "preference_only") and parsed_output.preference:
            p = parsed_output.preference
            if p.confidence >= self.min_confidence:
                if p.preferred_candidate in candidates and p.other_candidate in candidates:
                    x_pref = self._ensure_2d(candidates[p.preferred_candidate])
                    x_other = self._ensure_2d(candidates[p.other_candidate])
                    self.preference_history.append((x_pref, x_other))

        # 3. Aggiorna Vincoli
        if parsed_output.feedback_type in ("both", "direction_only"):
            valid_constraints = [
                c for c in parsed_output.constraints
                if c.confidence >= self.min_confidence
            ]
            self.constraint_specs.extend(valid_constraints)

        return parsed_output

    # -------------------------------------------------------------------------
    # 2) EXPORT PER PBO / BOTORCH
    # -------------------------------------------------------------------------

    def get_preference_dataset(self) -> List[Tuple[Tensor, Tensor]]:
        """Restituisce dataset (x_won, x_lost) per il training del modello di preferenza."""
        return self.preference_history

    def build_nonlinear_inequality_constraints(
        self,
        candidates_reference: Dict[str, Tensor],
    ) -> List[Tuple[Callable[[Tensor], Tensor], bool]]:
        """
        Genera vincoli callable compatibili con BoTorch `optimize_acqf`.
        Formato: (callable(X) -> Tensor >= 0, is_intrapoint=True)
        """
        constraints = []

        for spec in self.constraint_specs:
            if spec.subfunction_id not in self.sub_surrogates:
                continue

            J_model = self.sub_surrogates[spec.subfunction_id]
            threshold_val = self._calculate_threshold(spec, candidates_reference, J_model)

            if threshold_val is None:
                continue

            constraint_func = self._create_constraint_func(J_model, threshold_val, spec.operator)
            constraints.append((constraint_func, True))

        return constraints

    # -------------------------------------------------------------------------
    # 3) HELPERS
    # -------------------------------------------------------------------------

    def _calculate_threshold(
        self,
        spec: ConstraintSpec,
        candidates_reference: Dict[str, Tensor],
        J_model: Callable[[Tensor], Tensor]
    ) -> Optional[float]:
        """Calcola il valore numerico della soglia in base al tipo di vincolo."""

        if spec.constraint_type == "upper_bound_absolute":
            return spec.threshold

        if not spec.reference_candidate or spec.reference_candidate not in candidates_reference:
            return None

        x_ref = self._ensure_2d(candidates_reference[spec.reference_candidate])

        with torch.no_grad():
            val_ref = J_model(x_ref).item()

        if spec.constraint_type == "upper_bound_relative":
            return val_ref + spec.margin
        elif spec.constraint_type == "directional_improvement":
            return val_ref - spec.margin

        return None

    @staticmethod
    def _create_constraint_func(model: Callable, limit: float, operator: str = "<=") -> Callable[[Tensor], Tensor]:
        """
        Factory per creare la closure corretta.
        BoTorch vuole c(x) >= 0.
        Se il vincolo è Model(x) <= Limit -> Limit - Model(x) >= 0.
        """
        def constraint(X: Tensor) -> Tensor:
            batch_shape = X.shape[:-1]

            model_output = model(X)

            while model_output.dim() > len(batch_shape) and model_output.shape[-1] == 1:
                model_output = model_output.squeeze(-1)

            if model_output.dim() == 0:
                model_output = model_output.expand(batch_shape)
            elif model_output.shape != batch_shape:
                if model_output.numel() == X.shape[:-1].numel():
                    model_output = model_output.view(batch_shape)
                elif model_output.shape == batch_shape[1:]:
                    model_output = model_output.expand(batch_shape)

            if operator == '<=':
                return limit - model_output
            elif operator == ">=":
                return model_output - limit
            return limit - model_output

        return constraint

    @staticmethod
    def _ensure_2d(x: Tensor) -> Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x