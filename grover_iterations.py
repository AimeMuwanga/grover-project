"""
Grover's algorithm: iterations vs. success rate.

For several values of n, runs Grover with 0, 1, 2, ... iterations on the Aer
simulator, measures how often the marked state is found, and compares the
result with the theoretical curve  P(k) = sin^2((2k + 1) * theta),
where theta = arcsin(1 / sqrt(N)) and N = 2^n.
"""
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------- Grover building blocks ----------
def oracle(n, marked):
    """Flips the phase of the basis state `marked` (bitstring like '101')."""
    qc = QuantumCircuit(n)
    for i, b in enumerate(reversed(marked)):
        if b == "0":
            qc.x(i)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i, b in enumerate(reversed(marked)):
        if b == "0":
            qc.x(i)
    return qc


def diffuser(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover(n, marked, iterations=None):
    if iterations is None:
        iterations = round(math.pi / 4 * math.sqrt(2 ** n))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle(n, marked), inplace=True)
        qc.compose(diffuser(n), inplace=True)
    qc.measure_all()
    return qc


# ---------- Experiment ----------
def theory(n, k):
    theta = math.asin(1 / math.sqrt(2 ** n))
    return math.sin((2 * k + 1) * theta) ** 2


def measured_success(sim, n, marked, k, shots):
    qc = grover(n, marked, k)
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    return counts.get(marked, 0) / shots


def run_experiment(configs, shots=2048):
    sim = AerSimulator()
    results = {}
    for n, marked in configs:
        k_opt = round(math.pi / 4 * math.sqrt(2 ** n))
        k_max = 2 * k_opt + 2            # go well past the optimum
        ks = list(range(k_max + 1))
        rates = [measured_success(sim, n, marked, k, shots) for k in ks]
        results[(n, marked)] = (ks, rates, k_opt)

        print(f"\nn={n}, marked={marked}, optimal k={k_opt}")
        print(" k | measured | theory")
        for k, r in zip(ks, rates):
            print(f"{k:2d} |  {r:.3f}   | {theory(n, k):.3f}")
    return results


def plot_results(results, shots, filename="grover_iterations.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, ((n, marked), (ks, rates, k_opt)) in zip(axes, results.items()):
        fine_k = [i / 20 for i in range(int(max(ks) * 20) + 1)]
        ax.plot(fine_k, [theory(n, k) for k in fine_k], "-", label="theory")
        ax.plot(ks, rates, "o", label=f"simulation ({shots} shots)")
        ax.axvline(k_opt, linestyle="--", color="gray", label=f"optimal k = {k_opt}")
        ax.set_title(f"n = {n}, marked = {marked}")
        ax.set_xlabel("Grover iterations k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Success probability")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"\nSaved plot to {filename}")
    plt.show()


if __name__ == "__main__":
    SHOTS = 2048
    CONFIGS = [(3, "101"), (4, "1011"), (5, "10010")]
    results = run_experiment(CONFIGS, shots=SHOTS)
    plot_results(results, SHOTS)
