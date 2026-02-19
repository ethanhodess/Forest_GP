import argparse
from hashlib import new
import traceback
import random
import numpy as np
import pandas as pd
from collections import Counter
import ray
import copy
import dill as pickle
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


@ray.remote
def evaluate_individual(tree: DecisionTreeClassifier, X, y):
    y_pred = tree.predict(X)
    acc = (y_pred == y).mean()
    height = tree.get_depth()
    leaves = tree.get_n_leaves()
    return acc, height, leaves



DT_DEFAULTS = DecisionTreeClassifier().get_params(deep=False)

PARAM_SPACE = {
    "max_depth": {"type": "float"},
    "min_samples_split": {"type": "float", "min": 2},
    "min_samples_leaf": {"type": "float", "min": 1},
    "min_impurity_decrease": {"type": "float"},
    "criterion": {
        "type": "cat",
        "values": ["gini", "entropy", "log_loss"],
    },
    "max_features": {
        "type": "cat",
        "values": [None, "sqrt", "log2"],
    },
}

def sample_param(spec):
    if spec["type"] == "float":
        return random.random()
    if spec["type"] == "cat":
        return random.choice(spec["values"])
    raise ValueError("Unknown param type")



class GeneticProgrammingSystem:
    def __init__(self, pop_size, mutation_rate, tournament_k):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.population = []

    def _decode_params(self, gene_params, n_samples):
        decoded = {}

        for name, spec in PARAM_SPACE.items():
            gene = gene_params[name]

            if gene is None:
                decoded[name] = DT_DEFAULTS[name]
                continue

            if spec["type"] == "float":
                g = float(gene)
                if name == "min_samples_leaf":
                    # range: [1, 0.05 * n]
                    decoded[name] = max(
                        1,
                        int(1 + (g ** 2) * 0.05 * n_samples)
                    )

                elif name == "min_samples_split":
                    # range: [2, 0.2 * n]
                    decoded[name] = max(
                        2,
                        int(2 + (g ** 2) * 0.2 * n_samples)
                    )

                elif name == "max_depth":
                    # range: [1, 50]
                    decoded[name] = max(
                        1,
                        int(1 + (g ** 2) * 50)
                    )

                elif name == "min_impurity_decrease":
                    decoded[name] = g * 1e-2

            elif spec["type"] == "cat":
                decoded[name] = gene

        return decoded

    def initialize_population(self, X, y):
        self.population = []
        n = len(X)

        for _ in range(self.pop_size):
            gene_params = {k: None for k in PARAM_SPACE}

            for name, spec in PARAM_SPACE.items():
                gene_params[name] = self._mutate_gene(None, spec)

            tree = DecisionTreeClassifier()
            tree.gene_params_ = gene_params

            idx = np.random.choice(n, size=n, replace=True)
            tree.sample_indices_ = idx

            params = self._decode_params(gene_params, len(idx))
            tree.set_params(**params)
            tree.fit(X[idx], y[idx])

            self.population.append(tree)

    def _tournament_pick_index(self, fitnesses):
        competitors = random.sample(range(self.pop_size), self.tournament_k)
        return max(competitors, key=lambda i: fitnesses[i][0])

    def _mutate_gene(self, current, spec):
        # if default
        if current is None:
            if random.random() < self.mutation_rate:
                return sample_param(spec)
            return None

        # if non-default
        # coin flip to switch to default
        if random.random() < self.mutation_rate:
            return None

        # coin flip to mutate or stay same
        if random.random() < self.mutation_rate:
            if spec["type"] == "float":
                new = current + np.random.normal(0.0, 0.2) # +- 0.2
                return float(np.clip(new, 0.0, 1.0))
            else:
                return sample_param(spec)

        return current


    def _mutate_from_parent(self, parent):
        parent_genes = parent.gene_params_
        child_genes = {}

        bootstrap_mutate = random.random() < self.mutation_rate

        for name, spec in PARAM_SPACE.items():
            child_genes[name] = self._mutate_gene(
            parent.gene_params_[name], spec
        )

        child = DecisionTreeClassifier()
        child.gene_params_ = child_genes
        return child, bootstrap_mutate

    def evolve(self, X_train, y_train, X_val, y_val, gen_0=True):
        futures = [evaluate_individual.remote(t, X_val, y_val) for t in self.population]
        fitnesses = ray.get(futures)
        evaluated_population = copy.deepcopy(self.population)

        if gen_0:
            return fitnesses, evaluated_population

        n = len(X_train)
        new_population = []

        while len(new_population) < self.pop_size:
            parent_idx = self._tournament_pick_index(fitnesses)
            parent = self.population[parent_idx]

            child, bootstrap_mutate = self._mutate_from_parent(parent)

            if bootstrap_mutate:
                idx = np.random.choice(n, size=n, replace=True)
            else:
                idx = parent.sample_indices_

            child.sample_indices_ = idx

            params = self._decode_params(child.gene_params_, len(idx))
            child.set_params(**params)
            child.fit(X_train[idx], y_train[idx])

            new_population.append(child)

        self.population = new_population
        return fitnesses, evaluated_population



def majority_vote(preds_matrix):
    n_samples = preds_matrix.shape[1]
    votes = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        votes[i] = Counter(preds_matrix[:, i]).most_common(1)[0][0]
    return votes



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_jobs", default=30)
    parser.add_argument("-s", "--savepath", default="results_tables")
    parser.add_argument("-r", "--num_runs", default=1, nargs='?')
    args = parser.parse_args()

    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    TOURNAMENT_KS = [2, 10, 25, 50]

    try:

        ray.init(num_cpus=n_jobs, ignore_reinit_error=True, log_to_driver=False)

        task_ids = [359954, 2073, 190146, 168784, 359959]
        num_runs = 20
        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

        random.seed(run_num)
        np.random.seed(run_num)

        d = pickle.load(open(f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl', "rb"))
        X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
            random_state=run_num, stratify=y_train
        )

        print("\n=== Baseline: sklearn RandomForestClassifier ===")
        rf = RandomForestClassifier(
            n_estimators=5000,
            n_jobs=n_jobs,
            random_state=run_num
        )
        rf.fit(X_train, y_train)
        rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"RF test accuracy = {rf_test_acc:.4f}\n")

        rf_depth_var = np.var([t.get_depth() for t in rf.estimators_])
        rf_leaves_var = np.var([t.get_n_leaves() for t in rf.estimators_])
        rf_avg_score = np.mean([t.score(X_val, y_val) for t in rf.estimators_])

        for tournament_k in TOURNAMENT_KS:
            print(f"\n===== TOURNAMENT K = {tournament_k} =====")
            gp = GeneticProgrammingSystem(
                pop_size=100,
                mutation_rate=0.5,
                tournament_k=tournament_k
            )

            gp.initialize_population(X_train, y_train)
            cumulative_trees = []
            metrics = []

            for gen in range(50):
                fitnesses, evaluated = gp.evolve(
                    X_train, y_train, X_val, y_val, gen_0=(gen == 0)
                )

                cumulative_trees.extend(evaluated)

                preds_test = np.vstack([t.predict(X_test) for t in cumulative_trees])
                ens_test_acc = (majority_vote(preds_test) == y_test).mean()

                tree_test_accs = [
                    (t.predict(X_test) == y_test).mean()
                    for t in cumulative_trees
                ]

                heights = [t.get_depth() for t in cumulative_trees]
                leaves = [t.get_n_leaves() for t in cumulative_trees]

                metrics.append({
                    "run_id": task_id,
                    "run_num": run_num,
                    "tournament_k": tournament_k,
                    "generation": gen,
                    "avg_tree_test_acc": round(np.mean(tree_test_accs), 3),
                    "ensemble_test_acc": round(ens_test_acc, 3),
                    "height_var": round(np.var(heights), 3),
                    "leaves_var": round(np.var(leaves), 3),
                    "RF_baseline": round(rf_test_acc, 3),
                    "RF_height": round(rf_depth_var, 3),
                    "RF_leaves": round(rf_leaves_var, 3),
                    "RF_avg": round(rf_avg_score, 3),
                })
                print(
                    f"Gen {gen}: avg_tree={np.mean(tree_test_accs):.4f}, "
                    f"ens_test={ens_test_acc:.4f}"
                )

            pd.DataFrame(metrics).to_csv(
                f"{base_save_folder}/metrics_k{tournament_k}_{task_id}_{run_num}.csv",
                index=False
            )
    except Exception as e:
        trace = traceback.format_exc()
        print("Failed on ", base_save_folder)
        print(e)
        print(trace)


if __name__ == "__main__":
    main()
    print("DONE")
