import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import math


def oracle(n, marked):
    """Flips the phase of the basis state `marked` (a bitstring like '101')."""
    qc = QuantumCircuit(n)
    for i, b in enumerate(reversed(marked)):
        if b == '0':
            qc.x(i)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i, b in enumerate(reversed(marked)):
        if b == '0':
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


def optimal_iterations(n):
    theta = math.asin(1 / math.sqrt(2**n))
    return round(math.pi / (4 * theta) - 0.5)


def grover(n, marked, iterations=None):
    if iterations is None:
        iterations = optimal_iterations(n)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle(n, marked), inplace=True)
        qc.compose(diffuser(n), inplace=True)
    qc.measure_all()
    return qc


def run_examples(configs, shots=1024):
    sim = AerSimulator()
    for n, marked in configs:
        qc = grover(n, marked)
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        success_rate = counts.get(marked, 0) / shots
        filename = f"grover_n{n}_marked_{marked}.png"

        print(
            f"n={n}, marked={marked}, success rate={success_rate:.1%} "
            f"({counts.get(marked, 0)}/{shots})"
        )
        figure = plot_histogram(
            counts, title=f"Grover results: n={n}, marked={marked}"
        )
        figure.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved histogram to {filename}")


if __name__ == "__main__":
    CONFIGS = [
        (2, "00"),
        (2, "11"),
        (3, "101"),
        (3, "010"),
        (4, "1011"),
        (4, "0101"),
        (5, "10010"),
        (5, "01101"),
    ]
    run_examples(CONFIGS)