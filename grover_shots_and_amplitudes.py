"""
Grover's algorithm, part 2:
  (A) Shots convergence: measured success rate vs. number of shots,
      compared with the theoretical probability.
  (B) Amplitude picture: statevector amplitudes after each Grover iteration.
  (C) Bloch-sphere picture of the effective two-level system
      |rest> / |marked> (the Grover rotation).
"""
import math
import statistics
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector
from qiskit_aer import AerSimulator


# ---------- Grover building blocks ----------
def oracle(n, marked):
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


def grover(n, marked, iterations=None, measure=True):
    if iterations is None:
        iterations = round(math.pi / 4 * math.sqrt(2 ** n))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle(n, marked), inplace=True)
        qc.compose(diffuser(n), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def theta(n):
    return math.asin(1 / math.sqrt(2 ** n))


def theory(n, k):
    return math.sin((2 * k + 1) * theta(n)) ** 2


# ---------- (A) Shots convergence ----------
def shots_convergence(n=3, marked="101", shots_list=(10, 100, 1000, 10000),
                      repeats=50, filename="grover_shots_convergence.png"):
    sim = AerSimulator()
    k = round(math.pi / 4 * math.sqrt(2 ** n))
    p_theory = theory(n, k)
    qc = transpile(grover(n, marked, k), sim)

    means, stds = [], []
    print(f"\nShots convergence: n={n}, marked={marked}, k={k}, "
          f"theoretical P = {p_theory:.4f}")
    print(f"{'shots':>6} | {'mean':>7} | {'std':>7} | {'|mean - theory|':>15}")
    for shots in shots_list:
        # repeat the whole experiment to see the spread, not just one lucky run
        rates = []
        for _ in range(repeats):
            counts = sim.run(qc, shots=shots).result().get_counts()
            rates.append(counts.get(marked, 0) / shots)
        m, s = statistics.mean(rates), statistics.stdev(rates)
        means.append(m)
        stds.append(s)
        print(f"{shots:6d} | {m:7.4f} | {s:7.4f} | {abs(m - p_theory):15.4f}")

    plt.figure(figsize=(6, 4))
    plt.errorbar(shots_list, means, yerr=stds, fmt="o-", capsize=4,
                 label=f"measured (mean ± std, {repeats} repeats)")
    plt.axhline(p_theory, color="gray", linestyle="--",
                label=f"theory = {p_theory:.3f}")
    plt.xscale("log")
    plt.xlabel("Number of shots")
    plt.ylabel("Success probability")
    plt.title(f"Shots convergence (n={n}, marked={marked})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved {filename}")
    plt.show()


# ---------- (B) Amplitudes after each iteration ----------
def amplitude_picture(n=3, marked="101", filename="grover_amplitudes.png"):
    k_opt = round(math.pi / 4 * math.sqrt(2 ** n))
    ks = list(range(k_opt + 2))             # include one iteration past optimum
    labels = [format(i, f"0{n}b") for i in range(2 ** n)]

    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 3.2),
                             sharey=True)
    for ax, k in zip(axes, ks):
        sv = Statevector.from_instruction(grover(n, marked, k, measure=False))
        amps = [a.real for a in sv.data]
        colors = ["tab:red" if lab == marked else "tab:blue" for lab in labels]
        ax.bar(labels, amps, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"k = {k}")
        ax.tick_params(axis="x", rotation=90)
    axes[0].set_ylabel("Amplitude")
    fig.suptitle(f"Amplitudes after k Grover iterations (n={n}, marked={marked})")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved {filename}")
    plt.show()


# ---------- (C) Bloch picture of the effective two-level system ----------
def bloch_picture(n=3, k_values=None, prefix="grover_bloch"):
    """
    Grover's state always lies in the plane spanned by |marked> and the
    uniform superposition of all other states (|rest>). Treating these two as
    an effective qubit  cos(phi)|rest> + sin(phi)|marked>,  phi = (2k+1)*theta,
    gives a Bloch vector (sin 2phi, 0, cos 2phi): each Grover iteration rotates
    it by 2*theta around the y-axis, from near the |rest> pole (north) toward the |marked> pole (south).
    (This is an effective picture, not the Bloch sphere of one physical qubit.)
    """
    k_opt = round(math.pi / 4 * math.sqrt(2 ** n))
    if k_values is None:
        k_values = list(range(k_opt + 2))
    for k in k_values:
        phi = (2 * k + 1) * theta(n)
        vec = [math.sin(2 * phi), 0.0, math.cos(2 * phi)]
        fig = plot_bloch_vector(vec, title=f"k = {k}  (n={n})")
        fname = f"{prefix}_n{n}_k{k}.png"
        fig.savefig(fname, dpi=200)
        print(f"Saved {fname}")
    plt.show()


if __name__ == "__main__":
    shots_convergence(n=3, marked="101")
    amplitude_picture(n=3, marked="101")
    bloch_picture(n=3)
