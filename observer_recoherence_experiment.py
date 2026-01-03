#!/usr/bin/env python3
"""
observer_recoherence_experiment.py

Simulator: Aer qasm_simulator via backend.run()
Hardware: IBM Quantum via qiskit-ibm-runtime V2 primitives (SamplerV2)

Conditions:
  A) baseline:        H(S) -> H(S) -> measure S
  B) recohere:        H(S) -> record into k witnesses -> unrecord -> H(S) -> measure S
  C_mid) mid-measure: H(S) -> record -> MEASURE a witness mid-circuit -> attempt unrecord -> H(S) -> measure S

Classical bits:
  c[0] = witness measurement (only in C_mid)
  c[1] = system measurement (used in all)

Metric:
  p0 = P(system bit == 0), visibility V = 2*p0 - 1

Usage:

# (zsh) simulator only
python observer_recoherence_experiment.py

# (zsh) ~2-min hardware sanity run
RUN_HARDWARE=1 IBM_BACKEND=ibm_fez SHOTS=256 K_VALUES=1 python observer_recoherence_experiment.py

# (zsh) ~5-min hardware run
RUN_HARDWARE=1 IBM_BACKEND=ibm_fez SHOTS=256 K_VALUES=0,1,2,4 python observer_recoherence_experiment.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import Backend
from qiskit_aer import Aer

# IBM Runtime (primitives)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    IBM_AVAILABLE = True
except Exception:
    IBM_AVAILABLE = False


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class RunConfig:
    kmax: int = 9
    k_values: Optional[List[int]] = None

    shots: int = 2000
    seed: int = 7

    include_mid_measure: bool = True
    mid_measure_witness_idx: int = 1  # qubit index (1 = first witness)

    # Hardware
    run_hardware: bool = False
    ibm_backend_name: Optional[str] = None  # e.g. "ibm_fez"
    ibm_instance: Optional[str] = None      # e.g. "recoherence" (optional)


# ----------------------------
# Circuit builders
# ----------------------------

def build_baseline_circuit() -> QuantumCircuit:
    """A) Baseline interference: H -> H -> measure system into c[1]."""
    qc = QuantumCircuit(1, 2)
    S = 0
    qc.h(S)
    qc.h(S)
    qc.measure(S, 1)
    return qc


def build_recoherence_circuit(k: int) -> QuantumCircuit:
    """B) Coherent record + unrecord; measure system into c[1]."""
    n = 1 + k
    qc = QuantumCircuit(n, 2)
    S = 0

    qc.h(S)
    for i in range(1, n):
        qc.cx(S, i)
    for i in reversed(range(1, n)):
        qc.cx(S, i)

    qc.h(S)
    qc.measure(S, 1)
    return qc


def build_mid_measure_circuit(k: int, witness_qubit: int = 1) -> QuantumCircuit:
    """C_mid) True mid-circuit witness measurement into c[0], then continue."""
    n = 1 + k
    qc = QuantumCircuit(n, 2)
    S = 0

    if k <= 0:
        qc.h(S)
        qc.h(S)
        qc.measure(S, 1)
        return qc

    if witness_qubit < 1 or witness_qubit >= n:
        raise ValueError(f"witness_qubit must be in [1, {n-1}] but got {witness_qubit}")

    qc.h(S)
    for i in range(1, n):
        qc.cx(S, i)

    qc.barrier()
    qc.measure(witness_qubit, 0)
    qc.barrier()

    for i in reversed(range(1, n)):
        qc.cx(S, i)

    qc.h(S)
    qc.measure(S, 1)
    return qc


# ----------------------------
# Helpers
# ----------------------------

def parse_k_values(cfg: RunConfig) -> List[int]:
    if cfg.k_values:
        return sorted(set(cfg.k_values))
    return list(range(0, cfg.kmax + 1))


def p_system0_from_counts(counts: Dict[str, int], system_cbit: int = 1, ncbits: int = 2) -> float:
    """
    Compute P(system classical bit == 0).

    Qiskit bitstrings are ordered as c[ncbits-1]...c[0] (left->right).
    If system stored in c[1] with ncbits=2 => leftmost bit.
    """
    total = sum(counts.values())
    if total <= 0:
        return float("nan")

    pos = ncbits - 1 - system_cbit  # index from left

    ok = 0
    for bitstr, c in counts.items():
        s = bitstr.replace(" ", "")
        if len(s) != ncbits:
            continue
        if s[pos] == "0":
            ok += c
    return ok / total


def visibility_from_p0(p0: float) -> float:
    return 2.0 * p0 - 1.0


def aer_counts(backend: Backend, circuits: List[QuantumCircuit], shots: int, seed: int) -> List[Dict[str, int]]:
    tcircs = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=seed)
    job = backend.run(tcircs, shots=shots, seed_simulator=seed)
    res = job.result()
    return [res.get_counts(i) for i in range(len(circuits))]


def _extract_counts_from_pub_result(pub_result: Any, shots: int, ncbits: int = 2) -> Dict[str, int]:
    """
    Robustly extract counts from a SamplerV2 PubResult across minor API variations.

    Preferred:
      - pub_result.join_data().get_counts()    (combines registers; avoids needing the reg name)

    Common alternatives:
      - pub_result.data.<creg_name>.get_counts()   where <creg_name> is often "meas" or "c"
      - pub_result.data.quasi_dist                (older/other modes; convert to pseudo-counts)
    """
    # 0) Best: join all classical registers (works even if reg is named "c" instead of "meas")
    if hasattr(pub_result, "join_data"):
        try:
            jd = pub_result.join_data()
            if hasattr(jd, "get_counts"):
                counts = jd.get_counts()
                return {str(k).replace(" ", ""): int(v) for k, v in counts.items()}
        except Exception:
            # Fall through to other strategies
            pass

    data = getattr(pub_result, "data", None)
    if data is None:
        raise RuntimeError(f"Sampler pub_result has no .data attribute: {pub_result!r}")

    # 1) Try known/likely classical register fields first.
    #    - IBM docs often show `meas`
    #    - QuantumCircuit(n, 2) creates a classical register named `c`
    for reg_name in ("meas", "c"):
        reg = getattr(data, reg_name, None)
        if reg is not None:
            # Most common: BitArray.get_counts()
            if hasattr(reg, "get_counts"):
                counts = reg.get_counts()
                return {str(k).replace(" ", ""): int(v) for k, v in counts.items()}

            # If we only have per-shot bitstrings, build counts.
            if hasattr(reg, "get_bitstrings"):
                bitstrings = reg.get_bitstrings()
                counts: Dict[str, int] = {}
                for b in bitstrings:
                    s = str(b).replace(" ", "")
                    # normalize width if needed
                    if len(s) != ncbits and s.isdigit():
                        s = format(int(s), f"0{ncbits}b")
                    counts[s] = counts.get(s, 0) + 1
                return counts

    # 2) Generic: data behaves like a container; scan fields for a BitArray-like object
    #    (this covers cases where the reg name isn't meas/c).
    try:
        for name in getattr(data, "keys", lambda: [])():
            reg = getattr(data, str(name), None)
            if reg is None:
                continue
            if hasattr(reg, "get_counts"):
                counts = reg.get_counts()
                return {str(k).replace(" ", ""): int(v) for k, v in counts.items()}
            if hasattr(reg, "get_bitstrings"):
                bitstrings = reg.get_bitstrings()
                counts: Dict[str, int] = {}
                for b in bitstrings:
                    s = str(b).replace(" ", "")
                    if len(s) != ncbits and s.isdigit():
                        s = format(int(s), f"0{ncbits}b")
                    counts[s] = counts.get(s, 0) + 1
                return counts
    except Exception:
        pass

    # 3) Fallback: quasi distribution (some configurations / older outputs)
    qd = getattr(data, "quasi_dist", None)
    if qd is not None:
        counts: Dict[str, int] = {}
        for outcome, prob in qd.items():
            if isinstance(outcome, str):
                bits = outcome.replace(" ", "")
            else:
                bits = format(int(outcome), f"0{ncbits}b")
            counts[bits] = counts.get(bits, 0) + int(round(float(prob) * shots))
        if sum(counts.values()) == 0:
            for outcome, prob in qd.items():
                bits = outcome if isinstance(outcome, str) else format(int(outcome), f"0{ncbits}b")
                counts[bits] = max(1, int(float(prob) * shots))
        return counts

    raise RuntimeError(
        "Could not extract counts from SamplerV2 result. "
        f"Data keys: {list(getattr(data, 'keys', lambda: [])())}. "
        f"Data attrs (partial): {[a for a in dir(data) if a in ('meas','c','quasi_dist')]}. "
        "If you paste repr(pub_result) and repr(pub_result.data), I can pin the exact path."
    )


def ibm_counts(
    service: QiskitRuntimeService,
    backend_name: str,
    circuits: List[QuantumCircuit],
    shots: int,
    seed: int,
) -> List[Dict[str, int]]:
    """
    Run circuits on IBM hardware using SamplerV2 (job mode) and return counts dicts.
    """
    backend = service.backend(backend_name)
    tcircs = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=seed)

    # IMPORTANT: SamplerV2 init uses `mode=backend` in current IBM docs.
    sampler = SamplerV2(mode=backend)

    pubs = [(qc, []) for qc in tcircs]
    job = sampler.run(pubs=pubs, shots=shots)
    result = job.result()

    out: List[Dict[str, int]] = []
    for pub_result in result:
        out.append(_extract_counts_from_pub_result(pub_result, shots=shots, ncbits=2))
    return out


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    k_values_env = os.environ.get("K_VALUES", "").strip()
    k_values: Optional[List[int]] = None
    if k_values_env:
        k_values = [int(x.strip()) for x in k_values_env.split(",") if x.strip()]

    cfg = RunConfig(
        kmax=int(os.environ.get("KMAX", "9")),
        k_values=k_values,
        shots=int(os.environ.get("SHOTS", "2000")),
        seed=int(os.environ.get("SEED", "7")),
        include_mid_measure=os.environ.get("INCLUDE_MID", "1") == "1",
        mid_measure_witness_idx=int(os.environ.get("MID_WITNESS", "1")),
        run_hardware=os.environ.get("RUN_HARDWARE", "0") == "1",
        ibm_backend_name=os.environ.get("IBM_BACKEND", None),
        ibm_instance=os.environ.get("IBM_INSTANCE", None),
    )
    print(f"[cfg] {cfg}")

    ks = parse_k_values(cfg)
    print(f"[cfg] k sweep = {ks}")

    results: Dict[str, Dict[int, Dict[str, float]]] = {"ideal_sim": {}, "hardware": {}}

    # Simulator
    sim_backend = Aer.get_backend("qasm_simulator")
    for k in ks:
        circs = [build_baseline_circuit(), build_recoherence_circuit(k)]
        labels = ["A_baseline", "B_recohere"]
        if cfg.include_mid_measure:
            circs.append(build_mid_measure_circuit(k, witness_qubit=cfg.mid_measure_witness_idx))
            labels.append("C_mid_measure")

        counts_list = aer_counts(sim_backend, circs, shots=cfg.shots, seed=cfg.seed)
        row: Dict[str, float] = {}
        for lab, counts in zip(labels, counts_list):
            p0 = p_system0_from_counts(counts, system_cbit=1, ncbits=2)
            row[lab] = visibility_from_p0(p0)
        results["ideal_sim"][k] = row

    # Hardware
    if cfg.run_hardware:
        if not IBM_AVAILABLE:
            raise RuntimeError("Missing qiskit-ibm-runtime. pip install qiskit-ibm-runtime")
        if not cfg.ibm_backend_name:
            raise RuntimeError("Set IBM_BACKEND=... for hardware runs (e.g. ibm_fez).")

        # Setting instance explicitly silences warnings and locks the plan/instance
        if cfg.ibm_instance:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", instance=cfg.ibm_instance)
        else:
            service = QiskitRuntimeService(channel="ibm_quantum_platform")

        print(f"[ibm] selected backend: {cfg.ibm_backend_name}")

        for k in ks:
            circs = [build_baseline_circuit(), build_recoherence_circuit(k)]
            labels = ["A_baseline", "B_recohere"]
            if cfg.include_mid_measure:
                circs.append(build_mid_measure_circuit(k, witness_qubit=cfg.mid_measure_witness_idx))
                labels.append("C_mid_measure")

            counts_list = ibm_counts(
                service=service,
                backend_name=cfg.ibm_backend_name,
                circuits=circs,
                shots=cfg.shots,
                seed=cfg.seed,
            )

            row: Dict[str, float] = {}
            for lab, counts in zip(labels, counts_list):
                p0 = p_system0_from_counts(counts, system_cbit=1, ncbits=2)
                row[lab] = visibility_from_p0(p0)
            results["hardware"][k] = row

    # Save results
    out_path = "observer_recoherence_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ok] wrote {out_path}")

    # Plot
    def extract(series: Dict[int, Dict[str, float]], key: str) -> Tuple[List[int], List[float]]:
        xs, ys = [], []
        for k in ks:
            if k in series and key in series[k]:
                xs.append(k)
                ys.append(series[k][key])
        return xs, ys

    plt.figure()

    x, y = extract(results["ideal_sim"], "A_baseline")
    plt.plot(x, y, marker="o", label="ideal A baseline")
    x, y = extract(results["ideal_sim"], "B_recohere")
    plt.plot(x, y, marker="o", label="ideal B recohere")
    if cfg.include_mid_measure:
        x, y = extract(results["ideal_sim"], "C_mid_measure")
        plt.plot(x, y, marker="o", label="ideal C mid-measure")

    if results["hardware"]:
        x, y = extract(results["hardware"], "A_baseline")
        plt.plot(x, y, marker="s", label="hw A baseline")
        x, y = extract(results["hardware"], "B_recohere")
        plt.plot(x, y, marker="s", label="hw B recohere")
        if cfg.include_mid_measure:
            x, y = extract(results["hardware"], "C_mid_measure")
            plt.plot(x, y, marker="s", label="hw C mid-measure")

    plt.axhline(0.0)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("k (witness qubits)")
    plt.ylabel("visibility V = 2 P(system=0) - 1")
    plt.title("Recoherence vs witness size (ideal / hardware) + true mid-circuit measurement")
    plt.legend()
    plt.tight_layout()

    fig_path = "observer_recoherence_plot.png"
    plt.savefig(fig_path, dpi=160)
    print(f"[ok] wrote {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
