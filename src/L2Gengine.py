import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import torch
from torch import Tensor
from dotenv import load_dotenv

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
    """Output strutturato atteso dall'LLM."""
    feedback_type: Literal["both", "preference_only", "direction_only", "none"]
    preference: Optional[Preference] = None
    constraints: List[ConstraintSpec] = Field(default_factory=list)

# ====== PROMPT TEMPLATE =======================================================

L2G_SYSTEM_PROMPT = """
You are a Language-to-Guidance (L2G) engine for Preference-based Bayesian Optimization.
Your goal is to parse human feedback into mathematical constraints.

CONTEXT:
{context_json}

INSTRUCTIONS:
1. Analyze the user feedback comparing candidates (e.g., A vs B).
2. Extract global preferences (Who won?).
3. Extract constraints on specific subfunctions (metrics) defined in the CONTEXT.
4. Return pure JSON matching the provided schema.

IMPORTANT CLASSIFICATION RULES:
- If the user says "I prefer A" AND adds a condition (e.g., "but ensure overshoot < 5%"), set feedback_type to "both".
- If the user only gives a preference, use "preference_only".
- If the user only gives a constraint/instruction, use "direction_only".
"""

# ====== L2G ENGINE ============================================================

class L2GEngine:
    def __init__(
        self,
        context: Dict[str, Any],
        sub_surrogates: Dict[str, Callable[[Tensor], Tensor]],
        llm: Optional[ChatOpenAI] = None,
        min_confidence: float = 0.5,
    ):
        self.context = context
        self.sub_surrogates = sub_surrogates
        self.llm = llm
        self.min_confidence = min_confidence

        # State storage
        self.preference_history: List[Tuple[Tensor, Tensor]] = [] # (x_pref, x_other)
        self.constraint_specs: List[ConstraintSpec] = []

    # -------------------------------------------------------------------------
    # 1) PROCESS FEEDBACK
    # -------------------------------------------------------------------------

    def process_feedback(
        self,
        candidates: Dict[str, Tensor],
        feedback_text: str,
        simulated_output: Optional[Dict[str, Any]] = None,
    ):
        """
        Main entry point. Parse text -> Update internal state.
        """
        # 1. Ottieni output strutturato (da LLM o simulazione)
        if simulated_output:
            parsed_output = L2GOutput(**simulated_output)
        else:
            parsed_output = self._call_llm(feedback_text)

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

    def _call_llm(self, feedback_text: str) -> L2GOutput:
        """Invoca LLM usando Structured Output (metodo moderno)."""
        if not self.llm:
            raise ValueError("LLM instance not provided.")

        # Configura chain con output strutturato Pydantic
        structured_llm = self.llm.with_structured_output(L2GOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", L2G_SYSTEM_PROMPT),
            ("user", "{feedback_text}")
        ])
        chain = prompt | structured_llm
        
        response =  chain.invoke({
            "context_json": json.dumps(self.context, ensure_ascii=False),
            "feedback_text": feedback_text
        })

        print("LLM Parsed Output:", response)

        return response

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

            # Creiamo la funzione vincolo
            # Nota: threshold_val è float, J_model è callable
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

        # Per vincoli relativi, serve il candidato di riferimento (es. "A")
        if not spec.reference_candidate or spec.reference_candidate not in candidates_reference:
            return None

        x_ref = self._ensure_2d(candidates_reference[spec.reference_candidate])
        
        with torch.no_grad():
            val_ref = J_model(x_ref).item() # Assumiamo output scalare

        if spec.constraint_type == "upper_bound_relative":
            return val_ref + spec.margin
        elif spec.constraint_type == "directional_improvement":
            return val_ref - spec.margin # Deve essere "migliore di ref" (assumendo minimizzazione < val_ref)
        
        return None

    @staticmethod
    def _create_constraint_func(model: Callable, limit: float,operator: str = "<=") -> Callable[[Tensor], Tensor]:
        """
        Factory per creare la closure corretta.
        BoTorch vuole c(x) >= 0.
        Se il vincolo è Model(x) <= Limit -> Limit - Model(x) >= 0.
        """
        def constraint(X: Tensor) -> Tensor:
            # X can have shapes:
            # - [batch, d] during optimization
            # - [q, batch, d] for joint optimization
            
            # Determine the batch shape (all dims except the last one, d)
            batch_shape = X.shape[:-1]
            
            # Get model prediction (This tensor HAS gradients)
            model_output = model(X)  
            
            # --- FIX: Ensure shapes align using Tensor operations, NOT .item() ---
            
            # 1. Remove trailing singleton dimensions (e.g., [N, 1] -> [N])
            # Be careful not to squeeze if it's a scalar [1] and batch is scalar
            while model_output.dim() > len(batch_shape) and model_output.shape[-1] == 1:
                model_output = model_output.squeeze(-1)
            
            # 2. Handle Scalar outputs (e.g. output is [], batch is [q, batch])
            if model_output.dim() == 0:
                # Expand scalar to match batch shape
                model_output = model_output.expand(batch_shape)
                
            # 3. Handle Mismatches (Broadcast or Reshape)
            elif model_output.shape != batch_shape:
                # If we have [q*batch] but need [q, batch]
                if model_output.numel() == X.shape[:-1].numel():
                     model_output = model_output.view(batch_shape)
                # If we have [batch] but need [q, batch] (Broadcasting)
                elif model_output.shape == batch_shape[1:]: 
                     model_output = model_output.expand(batch_shape)
                else:
                    # Try automatic broadcasting as a last resort
                    pass

            # Calculate the constraint value
            # Since model_output tracks gradients, 'result' will also track gradients.
            if operator =='<=':
                return limit - model_output # >= 0  <=> model <= limit
            elif operator == ">=":
                return model_output - limit  # >= 0  <=> model >= limit
            return limit - model_output # Default case
        
        return constraint

    @staticmethod
    def _ensure_2d(x: Tensor) -> Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x