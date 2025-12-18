import argparse
import traceback
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional
import ray
import copy
import dill as pickle
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from DecisionTree import DecisionTree

warnings.filterwarnings("ignore")
random.seed(0)
np.random.seed(0)



# ==============  FITNESS EVALUATION  ==============

@ray.remote
def evaluate_individual(tree: DecisionTree, X, y):
    y_pred = tree.predict(X)
    acc = (y_pred == y).mean()
    height = tree.height()
    leaves = tree.num_leaves()
    return acc, height, leaves


# ==================  GP SYSTEM  ===================

class GeneticProgrammingSystem:
    def __init__(self, pop_size, n_features, mutation_rate, tournament_k, n_classes):
        self.pop_size = pop_size
        self.n_features = n_features
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.n_classes = n_classes
        self.population: list[DecisionTree] = []

        self.init_ranges = {
            "max_depth_pct": (0.01, 0.05),
            "min_samples_split_pct": (0.01, 0.1),
            "min_samples_leaf_pct": (0.001, 0.05),
            "min_impurity_decrease": (0.0, 0.02),
        }

    def _random_hyperparams(self):
        return {k: float(np.random.uniform(*v)) for k, v in self.init_ranges.items()}

    def initialize_population(self, X, y):
        self.population = []
        for _ in range(self.pop_size):
            p = self._random_hyperparams()
            ind = DecisionTree(
                max_depth_pct=p["max_depth_pct"],
                min_samples_split_pct=p["min_samples_split_pct"],
                min_samples_leaf_pct=p["min_samples_leaf_pct"],
                min_impurity_decrease=p["min_impurity_decrease"],
                n_features=self.n_features,
                n_classes=self.n_classes
            )
            ind.fit(X, y, use_indices=None)
            self.population.append(ind)

    def _tournament_pick_index(self, fitnesses, k=2):
        competitors = random.sample(range(self.pop_size), k)
        best = max(competitors, key=lambda idx: fitnesses[idx][0])
        return best

    def _mutate_from_parent(self, parent: DecisionTree):
        child = copy.deepcopy(parent)
        bootstrap_mutate = (random.random() < self.mutation_rate)

        def maybe_mutate_attr(attr_name, lower=None, upper=None, ensure_nonneg=True):
            if random.random() < self.mutation_rate:
                factor = np.random.normal(1.0, 0.1)
                old = getattr(child, attr_name)
                new = old * factor
                if lower is not None:
                    new = max(lower, new)
                if upper is not None:
                    new = min(upper, new)
                if ensure_nonneg:
                    new = max(0.0, new)
                setattr(child, attr_name, float(new))

        for attr in ["max_depth_pct", "min_samples_split_pct", "min_samples_leaf_pct", "min_impurity_decrease"]:
            maybe_mutate_attr(attr)

        return child, bootstrap_mutate

    def evolve(self, X, y, gen_0=True):
        evaluated_population = []

        if gen_0:
            for ind in self.population:
                ind.fit(X, y, use_indices=None)
        else:
            for ind in self.population:
                if ind.root is None:
                    ind.fit(X, y, use_indices=None)

        futures = [evaluate_individual.remote(ind, X, y) for ind in self.population]
        fitnesses = ray.get(futures)
        evaluated_population = [copy.deepcopy(ind) for ind in self.population]

        if gen_0:
            return fitnesses, evaluated_population

        new_population = []
        while len(new_population) < self.pop_size:
            parent_idx = self._tournament_pick_index(fitnesses, k=2)
            parent = self.population[parent_idx]
            child, bootstrap_changed = self._mutate_from_parent(parent)

            # Full new sample when mutating data
            if bootstrap_changed:
                child.fit(X, y, use_indices=None)
            else:
                if parent.sample_indices is not None:
                    child.fit(X, y, use_indices=parent.sample_indices)
                else:
                    child.fit(X, y, use_indices=None)

            new_population.append(child)

        self.population = new_population[:self.pop_size]

        return fitnesses, evaluated_population


# ====================  MAIN  =====================

def majority_vote(preds_matrix):
    n_samples = preds_matrix.shape[1]
    votes = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        votes[i] = Counter(preds_matrix[:, i]).most_common(1)[0][0]
    return votes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_jobs", default=30, nargs='?')
    parser.add_argument("-s", "--savepath", default="results_tables", nargs='?')
    parser.add_argument("-r", "--num_runs", default=1, nargs='?')
    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    base_save_folder = args.savepath
    num_runs = int(args.num_runs)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    try:
        ray.init(
            num_cpus=n_jobs,
            ignore_reinit_error=True,
            logging_level="ERROR",
            log_to_driver=False,
            _system_config={"metrics_report_interval_ms": 0}
        )

        file_path = '/common/hodesse/hpc_test/TPOT2_ensemble/data/2073_True.pkl'
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']



        print("\n=== Baseline: sklearn RandomForestClassifier ===")

        rf = RandomForestClassifier(
            n_estimators=10000,
            max_depth=None,
            n_jobs=n_jobs,
            bootstrap=True,
            random_state=task_id
        )
        rf.fit(X_train, y_train)
        rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"RF test accuracy  = {rf_test_acc:.4f}\n")

        n_classes = len(np.unique(y_train))
        gp = GeneticProgrammingSystem(
            pop_size=100,
            n_features=X_train.shape[1],
            mutation_rate=0.5,
            tournament_k=100,
            n_classes=n_classes
        )
        gp.initialize_population(X_train, y_train)

        cumulative_trees = []
        full_results = []

        n_gen = 100
        for gen in range(n_gen):
            gen0_flag = (gen == 0)
            fitnesses, evaluated_population = gp.evolve(X_train, y_train, gen_0=gen0_flag)

            # Add current generation's trees to cumulative set
            cumulative_trees.extend(copy.deepcopy(evaluated_population))

            # Compute ensemble predictions on cumulative trees
            preds_matrix_train = np.vstack([t.predict(X_train) for t in cumulative_trees])
            preds_matrix_test = np.vstack([t.predict(X_test) for t in cumulative_trees])
            ensemble_train_acc = (majority_vote(preds_matrix_train) == y_train).mean()
            ensemble_test_acc = (majority_vote(preds_matrix_test) == y_test).mean()

            # Individual tree metrics
            tree_test_accs = [(t.predict(X_test) == y_test).mean()for t in cumulative_trees]

            # Compute structural metrics over cumulative trees
            heights = [t.height() for t in cumulative_trees]
            leaves = [t.num_leaves() for t in cumulative_trees]

            print(f"Gen {gen}: cumulative_tree_avg={np.mean(tree_test_accs)}, "
                  f"height_var={np.var(heights):.2f}, leaves_var={np.var(leaves):.2f}, "
                  f"ensemble_train_acc={ensemble_train_acc:.4f}, ensemble_test_acc={ensemble_test_acc:.4f}")
            
            full_results.append({
                "run_id": task_id,
                "generation": gen,
                "avg_tree_test_acc": np.mean(tree_test_accs),
                "ensemble_train_acc": ensemble_train_acc,
                "ensemble_test_acc": ensemble_test_acc,
                "height_var": np.var(heights),
                "leaves_var": np.var(leaves),
            })

        df = pd.DataFrame(full_results)

        df = df.round({
            "avg_tree_train_acc": 4,
            "avg_tree_test_acc": 4,
            "ensemble_train_acc": 4,
            "ensemble_test_acc": 4,
            "height_var": 2,
            "leaves_var": 2,
        })

        csv_path = os.path.join(
            base_save_folder,
            f"gp100_forest_results_task{task_id}.csv"
        )

        df.to_csv(csv_path, index=False)

    except Exception as e:
        trace = traceback.format_exc()
        print("Failed on ", base_save_folder)
        print(e)
        print(trace)


if __name__ == "__main__":
    main()
    print("DONE")
