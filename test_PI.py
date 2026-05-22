import torch
import numpy as np
import matplotlib.pyplot as plt
import control
import warnings
import random
import time
from dotenv import load_dotenv

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qSimpleRegret, qExpectedImprovement, qUpperConfidenceBound,qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.acquisition.objective import LinearMCObjective

# LangChain / OpenAI
from langchain_openai import ChatOpenAI

# I tuoi script (Assumendo siano nella stessa cartella)
import control_utils as cu
from L2Gengine import L2GEngine

# Ignora warning sui tensori double/float
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================
# 1. CONFIGURAZIONE E SETUP
# ============================================================

load_dotenv()



def set_all_seeds(fixed_seed=42,seed_type="fixed"):
    """
    Imposta il seed per garantire la riproducibilità o per generare un seed casuale.
    
    Args:
        seed_type (str): "fixed" per seed fisso, "random" per seed casuale.
        fixed_seed (int): Seed fisso da usare (di default 42).
    """
    if seed_type == "fixed":
        seed = fixed_seed
        print(f"Using fixed seed: {seed}")
    elif seed_type == "random":
        seed = int(time.time()) % (2**31-1)  # Un seed casuale basato sul timestamp
        print(f"Using random seed: {seed}")
    else:
        raise ValueError("seed_type must be either 'fixed' or 'random'")

    # Imposta il seed per numpy, torch, e random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Se utilizzi la GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Garantisce che ogni GPU utilizzi lo stesso seed
        torch.backends.cudnn.deterministic = True  # Comportamento deterministico
        torch.backends.cudnn.benchmark = False  # Riduce la variabilità delle prestazioni


LLM_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bounds per Kp e Ki: [0.1, 0.1] a [5.0, 5.0]
BOUNDS = torch.tensor([[0.1, 0.001], [5.0, 3.0]], dtype=torch.double)

# Definizione del Plant (Sistema Fisico)
PLANT = cu.make_plant(wn=1.0, zeta=0.7) 

# Definizioni contesto per L2G
CONTEXT = {
    "task": "PI Controller Tuning for Step Response",
    "parameters": ["Kp", "Ki"],
    "subfunctions": ["overshoot_pct", "settling_time", "tracking_mse"],
    "subfunction_descriptions": {
        "overshoot_pct": "Percentage by which the response exceeds the reference (0-100). Physical_meaning: Measures stability and damping. High overshoot means the system is oscillatory, aggressive, or 'bumpy'. Low overshoot means the system is smooth or damped. Usually lower the better.",
        "settling_time": "Time in seconds to stay within 2% of reference. Physical_meaning: Measures speed of response. High settling time means the system is slow, sluggish, or lazy. Low settling time means the system is fast, snappy, or responsive. Usually lower the better.",
        "tracking_mse": "Mean Squared Error of tracking. Physical_meaning: General measure of error energy. High MSE indicates poor tracking or instability. Usually lower the better."
    }
}

SEED = 42

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def evaluate_candidate(x_tensor):
    kp = float(x_tensor[0, 0].item())
    ki = float(x_tensor[0, 1].item())
    
    # 1. Simula for LONGER than your constraint
    # If constraint is 15s, simulate for 30s or 40s
    T_MAX = 40.0
    t, y = cu.simulate_step(kp, ki, PLANT, t_final=T_MAX)
    
    # 2. Calcola Metriche
    metrics = cu.compute_metrics(t, y)
    
    # 3. Gestione NaN (Unstable or too slow)
    # The penalty must be distinctly worse than the constraint
    if np.isnan(metrics['settling_time']):
        metrics['settling_time'] = T_MAX + 10.0  # Assign 50.0, so it's clearly > 15.0
        
    # Optional: Penalize MSE if unstable
    if metrics['tracking_mse'] > 1e4 or np.isnan(metrics['tracking_mse']):
        metrics['tracking_mse'] = 1e5 

    return t, y, metrics

def update_surrogates(X_train, Y_dict):
    """
    Addestra GP separati per ogni metrica fisica.
    X_train: [N, 2]
    Y_dict: {'overshoot_pct': [N, 1], ...}
    Ritorna un dizionario di modelli.
    """
    models = {}
    for name, Y_data in Y_dict.items():
        # Normalizzazione Y semplice per stabilità numerica
        train_Y = (Y_data - Y_data.mean()) / (Y_data.std() + 1e-6)
        
        gp = SingleTaskGP(X_train, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models[name] = gp
    return models

# Wrapper per L2G: deve essere una funzione Callable(X) -> Tensor
def create_surrogate_wrapper(model, mean_std, std_std):
    """
    Crea una funzione che prende X [batch, d] o [q, batch, d] 
    e ritorna il valore DE-NORMALIZZATO con la shape corretta.
    """
    def wrapper(X):
        
        # Store original shape and ensure at least 2D
        original_shape = X.shape
        original_ndim = X.dim()
        
        # Convert to 2D: [n_samples, d]
        if X.dim() == 1:
            # [d] -> [1, d]
            X_2d = X.unsqueeze(0)
        elif X.dim() == 3:
            # [q, batch, d] -> [q*batch, d]
            X_2d = X.reshape(-1, X.shape[-1])
        else:
            # Already 2D [batch, d]
            X_2d = X
        
        # Get model prediction
        posterior = model.posterior(X_2d)
        mu_norm = posterior.mean  # [n, 1]
        
        # Denormalize: mu_real = mu_norm * std + mean
        val = mu_norm * std_std + mean_std
        
        # Reshape back to match input dimensions
        if original_ndim == 1:
            # Input was [d], return scalar or [1] 
            val = val.squeeze()
        elif original_ndim == 3:
            # Input was [q, batch, d], reshape from [q*batch, 1] to [q, batch]
            val = val.reshape(original_shape[0], original_shape[1])
        else:
            # Input was [batch, d], keep as [batch, 1] or squeeze to [batch]
            val = val.squeeze(-1)
        
        return val
    
    return wrapper


# Warm Start Function 
def warm_start_pbo(n_samples: int = 20):
    """
    Genera candidati casuali, li valuta tutti silenziosamente.
    Ritorna i dataset pronti per i surrogati e l'indice del migliore trovato.
    """
    print(f"--- WARM START: Simulating {n_samples} random controllers in background ---")
    
    # 1. Genera N campioni casuali nello spazio
    raw_x = draw_sobol_samples(bounds=BOUNDS, n=1, q=n_samples).squeeze(0).to(torch.double)
    
    # Storage temporaneo
    obs_metrics = {"overshoot_pct": [], "settling_time": [], "tracking_mse": []}
    obs_trajectories = [] 
    
    valid_indices = [] # Indici dei controller "decenti" da mostrare
    
    # 2. Valuta tutto (Silent Loop)
    for i in range(n_samples):
        # Feedback visivo minimo (progress bar testuale)
        if i % 5 == 0: print(f"Simulating {i}/{n_samples}...", end="\r")
        
        t, y, m = evaluate_candidate(raw_x[i].unsqueeze(0))
        
        # Salva metriche per i surrogati
        obs_metrics["overshoot_pct"].append(m["overshoot_pct"])
        obs_metrics["settling_time"].append(m["settling_time"])
        obs_metrics["tracking_mse"].append(m["tracking_mse"])
        obs_trajectories.append((t, y, m))
        
        # 3. Criterio di "Decenza" per l'utente
        # Esempio: Non esploso (MSE < 1000) e non piatto (MSE > 0.01)
        # Nota: Settling time 15.0 è il nostro cap per "non stabilizzato"
        if m['tracking_mse'] < 100.0 and m['settling_time'] < 14.0:
            valid_indices.append(i)

    print(f"\nDone. Found {len(valid_indices)} stable candidates out of {n_samples}.")
    
    # 4. Converti in tensori per BoTorch
    train_y_metrics = {k: torch.tensor(v).unsqueeze(-1).double() for k, v in obs_metrics.items()}
    
    # 5. Seleziona il punto di partenza "Best" per l'utente
    # Se abbiamo validi, prendiamo quello con MSE minore tra i validi.
    # Se non ne abbiamo, prendiamo il meno peggio tra tutti.
    if valid_indices:
        # Trova indice relativo in valid_indices che minimizza MSE
        valid_mses = [obs_metrics['tracking_mse'][i] for i in valid_indices]
        best_relative_idx = np.argmin(valid_mses)
        best_idx_global = valid_indices[best_relative_idx]
    else:
        # Fallback critico: prendi quello con MSE minimo assoluto
        print("WARNING: No stable controllers found. Check bounds/plant.")
        best_idx_global = np.argmin(obs_metrics['tracking_mse'])

    return raw_x, train_y_metrics, obs_metrics, obs_trajectories, best_idx_global

def check_real_constraints(candidate_metrics: dict, specs: list, ref_metrics_dict: dict):
    """
    Verifies if a candidate's REAL metrics satisfy the L2G constraints.
    
    Args:
        candidate_metrics: Dict of metrics for the new candidate (e.g., {'overshoot_pct': 12.5, ...})
        specs: List of ConstraintSpec objects from L2G.
        ref_metrics_dict: Dict containing metrics of reference candidates. 
                          Example: {'A': {'overshoot_pct': 10.0, ...}, 'B': {...}}
    
    Returns:
        (bool, str): (True, "OK") if valid, (False, "Reason") if invalid.
    """
    for spec in specs:
        # 1. Get the actual value we are checking
        val = candidate_metrics.get(spec.subfunction_id)
        
        # If the metric calculation failed (NaN) or is missing, we can't verify.
        if val is None or np.isnan(val):
            # Strict mode: fail if metric is missing
            return False, f"Metric {spec.subfunction_id} is NaN or missing."

        # 2. Determine the numerical limit
        limit = None
        
        if spec.constraint_type == "upper_bound_absolute":
            # Direct value: "Overshoot < 5%"
            if spec.threshold is None: continue # Should not happen for absolute, but safety first
            limit = spec.threshold
            
        elif spec.constraint_type in ["upper_bound_relative", "directional_improvement"]:
            # Relative value: "Overshoot < A's overshoot"
            ref_key = spec.reference_candidate # e.g., "A" or "B"
            
            if ref_key not in ref_metrics_dict:
                # Warning: We are trying to compare against a candidate we don't have metrics for.
                # This often happens if 'B' was the reference, but we are currently generating the NEW 'B'.
                # In that case, we might skip or use 'A' as fallback. Here we skip to be safe.
                print(f"  [Warn] Reference '{ref_key}' not found for validation. Skipping constraint.")
                continue
                
            ref_val = ref_metrics_dict[ref_key].get(spec.subfunction_id)
            
            if ref_val is None or np.isnan(ref_val):
                 return False, f"Reference {ref_key} has NaN {spec.subfunction_id}"

            # Apply Logic:
            # relative/directional usually implies: Reference Value +/- Margin
            if spec.operator == "<=":
                # e.g. "Better than A" (assuming lower is better) -> Limit = A_val - margin
                # e.g. "Same as A" -> Limit = A_val + margin (tolerance)
                # We trust the sign logic handled in L2G, but usually:
                limit = ref_val + spec.margin 
            elif spec.operator == ">=":
                limit = ref_val - spec.margin

        # 3. Check the Limit
        if limit is not None:
            if spec.operator == "<=" and val > limit:
                return False, f"{spec.subfunction_id} ({val:.2f}) > {limit:.2f} (Ref: {spec.reference_candidate})"
            
            elif spec.operator == ">=" and val < limit:
                return False, f"{spec.subfunction_id} ({val:.2f}) < {limit:.2f} (Ref: {spec.reference_candidate})"

    return True, "OK"

def optimize_acqf_discrete_filtered(
    acq_func, 
    bounds, 
    constraints_specs: list, 
    sub_surrogates: dict, 
    n_samples=5000
):
    """
    Genera candidati tramite campionamento denso, filtra quelli che violano
    i vincoli surrogati e seleziona il migliore in base all'AcqFunc.
    """
    # 1. Genera campioni densi nello spazio (Sobol è meglio di Random)
    # bounds: [2, d] -> candidati: [n_samples, 1, d] per compatibilità BoTorch
    candidates = draw_sobol_samples(bounds=bounds, n=n_samples, q=1).to(torch.double)
    
    # 2. Calcola la validità (Constraint Satisfaction) sui MODELLI (non simulazione)
    # Iniziamo assumendo tutti validi (mask True)
    valid_mask = torch.ones(n_samples, dtype=torch.bool)
    
    if constraints_specs:
        print(f"   Filtering {n_samples} candidates against {len(constraints_specs)} constraints...")
        
        # Iteriamo su ogni vincolo L2G
        for spec in constraints_specs:
            if spec.subfunction_id not in sub_surrogates:
                continue
                
            # Recuperiamo il modello e calcoliamo la predizione per TUTTI i punti
            model = sub_surrogates[spec.subfunction_id]
            
            # Nota: model(candidates) usa il wrapper creato in create_surrogate_wrapper
            # restituisce [n_samples, 1] o [n_samples]
            with torch.no_grad():
                preds = model(candidates) 
                
            # Assicuriamo dimensionalità corretta per confronto
            if preds.ndim > 1: preds = preds.squeeze()
            
            # Logica di filtraggio
            limit = spec.threshold  # Assumiamo vincoli assoluti per semplicità qui, ma va adattato
            
            # Se il vincolo è relativo o complesso, calcoliamo il limite
            # (Qui semplifico assumendo che spec.threshold sia già il numero target)
            # Se il threshold è None (es. relativo), va calcolato prima di chiamare questa fun
            if limit is None: continue 

            if spec.operator == "<=":
                # Mantieni solo quelli dove pred <= limit
                valid_mask = valid_mask & (preds <= limit)
            elif spec.operator == ">=":
                valid_mask = valid_mask & (preds >= limit)

    # 3. Applica il filtro
    valid_candidates = candidates[valid_mask]
    
    if valid_candidates.shape[0] == 0:
        print("   WARNING: No candidates satisfy surrogate constraints. Returning best unconstrained.")
        # Fallback: prendi tutto
        valid_candidates = candidates
    else:
        print(f"   Found {valid_candidates.shape[0]}/{n_samples} valid candidates in surrogate model.")

    # 4. Valuta Acquisition Function solo sui validi
    with torch.no_grad():
        # acq_func vuole [N, 1, d]
        acq_vals = acq_func(valid_candidates)
    
    # 5. Trova l'indice del migliore (Massimizzazione standard in BoTorch)
    best_idx = torch.argmax(acq_vals)
    best_candidate = valid_candidates[best_idx]
    
    return best_candidate, acq_vals[best_idx]

# ============================================================
# 3. MAIN LOOP PBO
# ============================================================

def main():
    print("=== STARTING PBO FOR PI CONTROL TUNING ===")
    
    set_all_seeds(seed_type='random')
    # --- A. Initialization ---
    train_x, train_y_metrics, obs_metrics, obs_trajectories, best_idx = warm_start_pbo(n_samples=1000)
    
    # Track the "B" from the previous iteration to allow referencing it later
    last_shown_candidate_x = None 
    last_shown_metrics = None

    # L2G Engine Init
    l2g = L2GEngine(context=CONTEXT, sub_surrogates={}, llm=LLM_MODEL)
    
    # --- B. Optimization Loop ---
    N_ITERATIONS = 5
    
    for iteration in range(N_ITERATIONS):
        print(f"\n\n=== ITERATION {iteration + 1}/{N_ITERATIONS} ===")
        
        # 1. Train Metric Surrogates
        print("Training Metric Surrogates...")
        metric_models = update_surrogates(train_x, train_y_metrics)
        
        # 2. Update L2G with new surrogates (wrapped)
        l2g_surrogates = {}
        for k, gp in metric_models.items():
            std_val = train_y_metrics[k].std() + 1e-6
            mean_val = train_y_metrics[k].mean()
            l2g_surrogates[k] = create_surrogate_wrapper(gp, mean_val, std_val)
            
        l2g.sub_surrogates = l2g_surrogates
        
        # 3. Train Preference Model
        pref_data = l2g.get_preference_dataset()
        pref_model = None
        
        if len(pref_data) > 0:
            print(f"Training Preference Model on {len(pref_data)} comparisons...")
            winners = [p[0] for p in pref_data]
            losers = [p[1] for p in pref_data]
            
            X_win = torch.cat(winners, dim=0)
            X_lose = torch.cat(losers, dim=0)
            train_X = torch.cat([X_win, X_lose], dim=0)
            
            M = X_win.shape[0]
            comps = torch.tensor([[i, M + i] for i in range(M)], dtype=torch.long)
            
            pref_model = PairwiseGP(datapoints=train_X, comparisons=comps)
            mll_pref = PairwiseLaplaceMarginalLogLikelihood(pref_model.likelihood, pref_model)
            fit_gpytorch_mll(mll_pref)

        # 4. Build Constraints from Language
        # We construct a reference dict containing A (current best) AND B (previous)
        cand_ref = {"A": train_x[best_idx]} 
        if last_shown_candidate_x is not None:
            cand_ref["B"] = last_shown_candidate_x.squeeze(0)

        constraints_funcs = l2g.build_nonlinear_inequality_constraints(candidates_reference=cand_ref)
        
        if constraints_funcs:
            print(f"Applying {len(constraints_funcs)} active language constraints!")

        # 5. ROBUST OPTIMIZATION LOOP
        print("Optimizing Acquisition Function...")
        
        found_valid_candidate = False
        max_retries = 10
        xA = train_x[best_idx]
        # Prepare Objective for Cold Start (Minimize MSE -> Maximize -MSE)
        # We assume we are optimizing the first metric model (MSE usually)
        mse_model = metric_models["tracking_mse"]

        # We need to flip MSE because BoTorch maximizes. 
        # Since we normalized Y, we can just minimize the normalized output or maximize negative.
        # Let's use a generic approach: Best observed normalized Y so far (minimized)
        y_norm_samples = mse_model.train_targets
        best_f = y_norm_samples.min() # Best observed (normalized) MSE

        
        # --- FIX 1: Use UCB for Exploration instead of SimpleRegret ---
        if pref_model:
            # Beta=3.0 encourages exploring uncertain regions (avoids getting stuck at 'Best')
            acq_func = qUpperConfidenceBound(pref_model, beta=3.0)
        else:
            # Cold Start: Use Expected Improvement on MSE
            # We want to minimize MSE. BoTorch maximizes. 
            # We use a LinearMCObjective to negate the output: f(x) = -1 * output
            negate_obj = LinearMCObjective(weights=torch.tensor([-1.0], dtype=torch.double))
            
            # We want to IMPROVE upon the best_f (which is the minimum MSE).
            # Since we negate, the 'best' is -best_f.
            acq_func = qLogExpectedImprovement(
                model=mse_model, 
                best_f=-best_f, 
                objective=negate_obj
            )

        # Prepare Reference Metrics for Verification (The "Real" Check)
        # We need these to verify relative constraints like "better than A"
        metricsA = obs_trajectories[best_idx][2]
        ref_metrics_dict = {"A": metricsA}
        if last_shown_metrics is not None:
            ref_metrics_dict["B"] = last_shown_metrics

        for attempt in range(max_retries):
            # A. Find Candidate
            try:
                new_candidate, _ = optimize_acqf(
                    acq_function=acq_func, 
                    bounds=BOUNDS,
                    q=1,
                    num_restarts=10,
                    raw_samples=2048,
                    nonlinear_inequality_constraints=constraints_funcs if constraints_funcs else None,
                    ic_generator=gen_batch_initial_conditions,
                )
            except RuntimeError:
                print(f"  -> Attempt {attempt+1}: Optimization failed (constraints too strict). Fallback...")
                new_candidate = draw_sobol_samples(bounds=BOUNDS, n=1, q=1).squeeze(0).to(torch.double)
            except Exception as e:
                print(f"  -> Optimizer error: {e}")
                new_candidate = draw_sobol_samples(bounds=BOUNDS, n=1, q=1).squeeze(0).to(torch.double)

            # B. Simulate (Reality Check)
            tB, yB, metricsB = evaluate_candidate(new_candidate)
            
            # --- FIX 2: Check Real Constraints using Reference Dict ---
            is_valid, reason = check_real_constraints(metricsB, l2g.constraint_specs, ref_metrics_dict)
            
            if is_valid:
                found_valid_candidate = True
                print(f"  -> Attempt {attempt+1}: Valid candidate found! ({reason})")
                break 
            else:
                print(f"  -> Attempt {attempt+1}: Failed verification ({reason}). Retrying...")
                
                # C. Active Learning (Add failed point -> Refit -> Retry)
                train_x = torch.cat([train_x, new_candidate], dim=0)
                for k in obs_metrics:
                    val = torch.tensor([[metricsB[k]]], dtype=torch.double)
                    train_y_metrics[k] = torch.cat([train_y_metrics[k], val], dim=0)
                    obs_metrics[k].append(metricsB[k])
                
                obs_trajectories.append((tB, yB, metricsB))
                
                # Re-fit
                metric_models = update_surrogates(train_x, train_y_metrics)
                
                # Update L2G wrappers
                l2g_surrogates = {}
                for k, gp in metric_models.items():
                    std_val = train_y_metrics[k].std() + 1e-6
                    mean_val = train_y_metrics[k].mean()
                    l2g_surrogates[k] = create_surrogate_wrapper(gp, mean_val, std_val)
                l2g.sub_surrogates = l2g_surrogates
                
                # Re-build Constraints (cand_ref remains valid)
                constraints_funcs = l2g.build_nonlinear_inequality_constraints(candidates_reference=cand_ref)

        if not found_valid_candidate:
            print("WARNING: Could not find a candidate satisfying all constraints.")
            
        # 6. Setup Visualization
        tA, yA, metricsA = obs_trajectories[best_idx]
        xA = train_x[best_idx]
        xB = new_candidate.squeeze(0)

        # 7. Visualization
        print("Displaying comparison...")
        cu.plot_two_responses(
            tA, yA, metricsA, xA.numpy(),
            tB, yB, metricsB, xB.numpy()
        )
        
        # 8. Human Feedback
        print("\n--- HUMAN FEEDBACK ---")
        user_text = input("Your feedback (e.g. 'I prefer B', 'Make overshoot like A'): ")
        
        # 9. Process Feedback
        # --- FIX 3: Ensure L2G sees BOTH candidates for context ---
        candidates_dict = {"A": xA, "B": xB}
        
        # MODIFICA: Catturiamo l'oggetto ritornato (che contiene 'preference')
        parsed_output = l2g.process_feedback(candidates_dict, user_text)
        
        # Update Training Data
        if not torch.equal(train_x[-1], new_candidate):
            train_x = torch.cat([train_x, new_candidate], dim=0)
            for k in obs_metrics:
                val = torch.tensor([[metricsB[k]]], dtype=torch.double)
                train_y_metrics[k] = torch.cat([train_y_metrics[k], val], dim=0)
                obs_metrics[k].append(metricsB[k])
            obs_trajectories.append((tB, yB, metricsB))
        
        # Identifichiamo gli indici attuali
        idx_A = best_idx                # L'indice del vecchio migliore
        idx_B = len(train_x) - 1        # L'indice dell'ultimo aggiunto (il nuovo B)
        
        # Estraiamo la label preferita in modo sicuro
        preferred_label = None
        feedback_type = "none"
        
        if parsed_output:
            feedback_type = parsed_output.feedback_type
            # CONTROLLO CRITICO: Verifichiamo se 'preference' esiste prima di accedere
            if parsed_output.preference:
                preferred_label = parsed_output.preference.preferred_candidate
        
        # Logica decisionale robusta
        if preferred_label == 'B':
            best_idx = idx_B
            print(f"--> System updated: Baseline switched to B (User chose B).")
            
        elif preferred_label == 'A':
            best_idx = idx_A
            print(f"--> System updated: Baseline remains A (User chose A).")
            
        else:
            # Caso: "direction_only" (es. "Reduce overshoot") o "none"
            # Se non c'è un vincitore chiaro, manteniamo la baseline attuale (Approccio Conservativo)
            best_idx = idx_A
            
            # Forniamo feedback utile all'utente su cosa è successo
            num_constraints = len(parsed_output.constraints) if parsed_output else 0
            if num_constraints > 0:
                print(f"--> System updated: Baseline remains A, but {num_constraints} new constraints were extracted.")
            else:
                print(f"--> System updated: Baseline remains A (No clear preference or constraints detected).")


    
    print("\n=== PBO OPTIMIZATION COMPLETE ===")
    print(f"Final best controller: Kp={train_x[best_idx][0]:.3f}, Ki={train_x[best_idx][1]:.3f}")
    # Print final metrics
    final_metrics = obs_trajectories[best_idx][2]
    print(f"Final Metrics: Overshoot={final_metrics['overshoot_pct']:.2f}%, Settling Time={final_metrics['settling_time']:.2f}s, MSE={final_metrics['tracking_mse']:.4f}")

if __name__ == "__main__":
    main()