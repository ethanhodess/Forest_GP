import argparse
import traceback
import random
import numpy as np
from collections import Counter
import ray
import copy
import dill as pickle
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from DecisionTree import DecisionTree

warnings.filterwarnings("ignore")


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
random.seed(task_id)
np.random.seed(task_id)



# ==============  TREE METRICS EVALUATION  =============

@ray.remote
def evaluate_tree_metrics(tree: DecisionTree, X, y):
    y_pred = tree.predict(X)
    acc = (y_pred == y).mean()
    height = tree.height()
    leaves = tree.num_leaves()
    return (acc, height, leaves)


# ==================  GP SYSTEM  ===================

class GPSystem:
    def __init__(self, pop_size, n_features, mutation_rate, tournament_k, n_classes):
        self.pop_size = pop_size
        self.n_features = n_features
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.n_classes = n_classes
        self.population: list[DecisionTree] = []

        # initial ranges for hyperparams (fractions)
        self.init_ranges = {
            "max_depth_pct": (0.01, 0.05),
            "min_samples_split_pct": (0.01, 0.1),
            "min_samples_leaf_pct": (0.001, 0.05),
            "min_impurity_decrease": (0.0, 0.02),
            "bootstrap_pct": (0.5, 1.0)
        }

    def _random_hyperparams(self):
        return {
            "max_depth_pct": float(np.random.uniform(*self.init_ranges["max_depth_pct"])),
            "min_samples_split_pct": float(np.random.uniform(*self.init_ranges["min_samples_split_pct"])),
            "min_samples_leaf_pct": float(np.random.uniform(*self.init_ranges["min_samples_leaf_pct"])),
            "min_impurity_decrease": float(np.random.uniform(*self.init_ranges["min_impurity_decrease"])),
            "bootstrap_pct": float(np.random.uniform(*self.init_ranges["bootstrap_pct"]))
        }

    def initialize_population(self, X, y):
        self.population = []
        for _ in range(self.pop_size):
            p = self._random_hyperparams()
            ind = DecisionTree(
                max_depth_pct=p["max_depth_pct"],
                min_samples_split_pct=p["min_samples_split_pct"],
                min_samples_leaf_pct=p["min_samples_leaf_pct"],
                min_impurity_decrease=p["min_impurity_decrease"],
                bootstrap_pct=p["bootstrap_pct"],
                n_features=self.n_features,
                n_classes=self.n_classes
            )
            # draw initial bootstrap sample and fit (this sets ind.sample_indices)
            ind.fit(X, y, use_indices=None)
            self.population.append(ind)

    # tournament pick by accuracy stored in fitnesses; returns index
    def _tournament_pick_index(self, fitnesses, k=2):
        competitors = random.sample(range(self.pop_size), k)
        best = max(competitors, key=lambda idx: fitnesses[idx][0])
        return best

    # mutate hyperparams copy of parent -> returns mutated child (deepcopy) and flag bootstrap_pct_changed (True/False)
    def _mutate_from_parent(self, parent: DecisionTree):
        child = copy.deepcopy(parent)
        bootstrap_changed = False

        # flip a coin for each hyperparameter: if coin says mutate -> multiply by Normal(1,0.1)
        def maybe_mutate_attr(attr_name, lower=None, upper=None, ensure_nonneg=True):
            nonlocal bootstrap_changed
            if random.random() < self.mutation_rate:
                factor = np.random.normal(1.0, 0.1)
                old = getattr(child, attr_name)
                new = old * factor
                # clip to bounds if provided
                if lower is not None:
                    new = max(lower, new)
                if upper is not None:
                    new = min(upper, new)
                if ensure_nonneg:
                    new = max(0.0, new)
                setattr(child, attr_name, float(new))
                if attr_name == "bootstrap_pct":
                    bootstrap_changed = True

        # mutate each hyperparameter independently
        maybe_mutate_attr("max_depth_pct", lower=1e-4, upper=1.0)
        maybe_mutate_attr("min_samples_split_pct", lower=1e-4, upper=1.0)
        maybe_mutate_attr("min_samples_leaf_pct", lower=1e-5, upper=1.0)
        maybe_mutate_attr("min_impurity_decrease", lower=0.0, upper=1.0)
        maybe_mutate_attr("bootstrap_pct", lower=0.0, upper=1.0)

        return child, bootstrap_changed


    def evolve(self, X, y, bootstrap=True):
        evaluated_population = []

        # ensure all individuals are fitted
        if bootstrap:
            for ind in self.population:
                ind.fit(X, y, use_indices=None)
        else:
            for ind in self.population:
                if ind.root is None:
                    ind.fit(X, y, use_indices=None)

        # Evaluate current population (use their fitted trees)
        futures = [evaluate_tree_metrics.remote(ind, X, y) for ind in self.population]
        fitnesses = ray.get(futures)

        # deep-copy evaluated population for returning (predictions later use evaluated_population)
        evaluated_population = [copy.deepcopy(ind) for ind in self.population]

        # If generation 0, skip reproduction
        if bootstrap:
            return fitnesses, evaluated_population

        # Otherwise, create new_population via mutation-only reproduction using tournament selection (k=2)
        new_population = []

        while len(new_population) < self.pop_size:
            # pick parent via tournament k=2
            parent_idx = self._tournament_pick_index(fitnesses, k=2)
            parent = self.population[parent_idx]

            # mutate hyperparams to produce child
            child, bootstrap_changed = self._mutate_from_parent(parent)

            # determine child's sample indices:
            # - if bootstrap_changed: keep 90% of parent's indices and resample 10% from full dataset (with replacement)
            # - else: inherit parent's indices directly (parent passes sample)
            if bootstrap_changed:
                if parent.sample_indices is None:
                    # parent had no bootstrap: draw fresh according to child's bootstrap_pct
                    child.fit(X, y, use_indices=None)
                else:
                    parent_idxs = np.array(parent.sample_indices, dtype=int)
                    n = len(parent_idxs)
                    # exactly 10% replacement (at least 1 replacement when n>0)
                    k_replace = max(1, int(round(0.10 * n)))
                    keep_k = max(0, n - k_replace)
                    # randomly pick which positions to keep
                    if keep_k > 0:
                        keep_positions = np.random.choice(n, keep_k, replace=False)
                        kept = parent_idxs[keep_positions]
                    else:
                        kept = np.array([], dtype=int)
                    # resample replacements from full data (indices)
                    replacements = np.random.choice(len(X), k_replace, replace=True) if k_replace > 0 else np.array([], dtype=int)
                    child_idxs = np.concatenate([kept, replacements]).astype(int)
                    np.random.shuffle(child_idxs)
                    # fit child with new indices (this sets child.sample_indices)
                    child.fit(X, y, use_indices=child_idxs)
            else:
                # inherit parent's indices (if parent has them) else let child draw sample according to its bootstrap_pct
                if parent.sample_indices is not None:
                    child.fit(X, y, use_indices=parent.sample_indices)
                else:
                    child.fit(X, y, use_indices=None)

            new_population.append(child)

        # truncate to pop_size (in case overshot)
        self.population = new_population[:self.pop_size]

        # return fitnesses computed earlier and deep-copied evaluated_population
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

        # Baseline: sklearn RandomForestClassifier (seeded by task_id) 
        print("\n=== Baseline: sklearn RandomForestClassifier ===")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            n_jobs=n_jobs,
            bootstrap=True,
            random_state=task_id
        )
        rf.fit(X_train, y_train)
        rf_test_acc  = accuracy_score(y_test, rf.predict(X_test))
        print(f"RF test accuracy  = {rf_test_acc:.4f}\n")


        # GP System
        n_classes = len(np.unique(y_train))
        gp = GPSystem(
            pop_size=100,
            n_features=X_train.shape[1],
            mutation_rate=0.5,       
            tournament_k=3,          
            n_classes=n_classes
        )
        gp.initialize_population(X_train, y_train)

        n_gen = 100
        for gen in range(n_gen):
            # Bagging only in generation 0 (matching your original main semantics)
            bootstrap_flag = (gen == 0)
            fitnesses, evaluated_population = gp.evolve(X_train, y_train, bootstrap=bootstrap_flag)

            accs = [f[0] for f in fitnesses]
            heights = [f[1] for f in fitnesses]
            leaves = [f[2] for f in fitnesses]

            preds_matrix_train = np.vstack([t.predict(X_train) for t in evaluated_population])
            preds_matrix_test  = np.vstack([t.predict(X_test) for t in evaluated_population])
            ensemble_train_acc = (majority_vote(preds_matrix_train) == y_train).mean()
            ensemble_test_acc  = (majority_vote(preds_matrix_test) == y_test).mean()

            print(f"Gen {gen}: avg_acc={np.mean(accs):.4f}, leaves_var={np.var(leaves):.2f}, "
                  f"height_var={np.var(heights):.2f} | "
                  f"ensemble_train_acc={ensemble_train_acc:.4f}, ensemble_test_acc={ensemble_test_acc:.4f}")

    except Exception as e:
        trace = traceback.format_exc()
        print("Failed on ", base_save_folder)
        print(e)
        print(trace)

if __name__ == "__main__":
    main()
    print("DONE")
