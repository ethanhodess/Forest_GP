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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from DecisionTree import DecisionTree

warnings.filterwarnings("ignore")


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
            "max_depth_gene": (0, 1),
            "min_samples_split_gene": (0, 1),
            "min_samples_leaf_gene": (0, 1),
            "min_impurity_gene": (0, 1),
        }

    def _random_hyperparams(self):
        return {k: float(np.random.uniform(*v)) for k, v in self.init_ranges.items()}

    def initialize_population(self, X, y):
        self.population = []
        for _ in range(self.pop_size):
            p = self._random_hyperparams()
            ind = DecisionTree(
                max_depth_gene=p["max_depth_gene"],
                min_samples_split_gene=p["min_samples_split_gene"],
                min_samples_leaf_gene=p["min_samples_leaf_gene"],
                min_impurity_gene=p["min_impurity_gene"],
                n_features=self.n_features,
                n_classes=self.n_classes,
                max_features=random.choice(DecisionTree._allowed_max_features),
                class_weight=random.choice(DecisionTree._allowed_class_weights),
                criterion=random.choice(DecisionTree._allowed_criteria)
            )
            ind.fit(X, y, use_indices=None)
            self.population.append(ind)

    def _tournament_pick_index(self, fitnesses):
        k = self.tournament_k
        competitors = random.sample(range(self.pop_size), k)
        best = max(competitors, key=lambda idx: fitnesses[idx][0])
        return best

    def _mutate_from_parent(self, parent: DecisionTree):
        child = copy.deepcopy(parent)
        bootstrap_mutate = (random.random() < self.mutation_rate)

        def maybe_mutate_attr(attr_name):
            if random.random() < self.mutation_rate:
                old = getattr(child, attr_name)
                new = np.clip(old + np.random.normal(0.0, 0.1), 0.0, 1.0)  # +/- 0.1, within [0, 1]
                setattr(child, attr_name, float(new))

        for attr in [
            "max_depth_gene",
            "min_samples_split_gene",
            "min_samples_leaf_gene",
            "min_impurity_gene",
        ]:
            maybe_mutate_attr(attr)

        if random.random() < self.mutation_rate:
            child.criterion = random.choice(child._allowed_criteria)

        if random.random() < self.mutation_rate:
            child.max_features = random.choice(child._allowed_max_features)

        if random.random() < self.mutation_rate:
            child.class_weight = random.choice(child._allowed_class_weights)

        return child, bootstrap_mutate

    def evolve(self, X_train, y_train, X_val, y_val, gen_0=True):
        if gen_0:
            for ind in self.population:
                ind.fit(X_train, y_train, use_indices=None)
        else:
            for ind in self.population:
                if ind.root is None:
                    ind.fit(X_train, y_train, use_indices=None)


        futures = [evaluate_individual.remote(ind, X_val, y_val) for ind in self.population]
        fitnesses = ray.get(futures)
        evaluated_population = [copy.deepcopy(ind) for ind in self.population]

        if gen_0:
            return fitnesses, evaluated_population

        new_population = []
        while len(new_population) < self.pop_size:
            parent_idx = self._tournament_pick_index(fitnesses)
            parent = self.population[parent_idx]
            child, bootstrap_changed = self._mutate_from_parent(parent)

            if bootstrap_changed:
                child.fit(X_train, y_train, use_indices=None)
            else:
                if parent.sample_indices is not None:
                    child.fit(X_train, y_train, use_indices=parent.sample_indices)
                else:
                    child.fit(X_train, y_train, use_indices=None)

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

    TOURNAMENT_KS = [1, 2, 10, 25, 50, 100]

    try:
        ray.init(
            num_cpus=n_jobs,
            ignore_reinit_error=True,
            logging_level="ERROR",
            log_to_driver=False,
            _system_config={"metrics_report_interval_ms": 0}
        )

        task_ids = [359954, 2073, 190146, 168784, 359959]
        num_runs = 20
        jobs = [(tid, run) for tid in task_ids for run in range(num_runs)]

        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_id, run_num = jobs[array_id]

        random.seed(run_num)
        np.random.seed(run_num)

        file_path = f'/common/hodesse/hpc_test/TPOT2_ensemble/data/{task_id}_True.pkl'
        d = pickle.load(open(file_path, "rb"))
        X_train, y_train, X_test, y_test = (
            d['X_train'], d['y_train'], d['X_test'], d['y_test']
        )

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=run_num, stratify=y_train)


        print("\n=== Baseline: sklearn RandomForestClassifier ===")

        rf = RandomForestClassifier(
            n_estimators=5000,
            max_depth=None,
            n_jobs=n_jobs,
            bootstrap=True,
            random_state=run_num
        )
        rf.fit(X_train, y_train)
        rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"RF test accuracy = {rf_test_acc:.4f}\n")

        rf_depth = []
        for tree in rf.estimators_:
            rf_depth.append(tree.get_depth())

        rf_depth_var = np.var(rf_depth)

        rf_leaves = []
        for tree in rf.estimators_:
            rf_leaves.append(tree.get_n_leaves())

        rf_leaves_var = np.var(rf_leaves)

        ind_tree_scores = []
        for tree in rf.estimators_:
            ind_tree_scores.append(tree.score(X_val, y_val))

        rf_avg_score = np.mean(ind_tree_scores)
        

        n_classes = len(np.unique(y_train))

        for tournament_k in TOURNAMENT_KS:
            print(f"\n===== TOURNAMENT K = {tournament_k} =====")

            gp = GeneticProgrammingSystem(
                pop_size=100,
                n_features=X_train.shape[1],
                mutation_rate=0.5,
                tournament_k=tournament_k,
                n_classes=n_classes
            )
            gp.initialize_population(X_train, y_train)

            tree_records = []
            metrics_records = []
            tree_id_counter = 0
            cumulative_trees = []

            n_gen = 50
            for gen in range(n_gen):
                gen0_flag = (gen == 0)
                fitnesses, evaluated_population = gp.evolve(
                    X_train, y_train, X_val, y_val, gen_0=gen0_flag
                )

                cumulative_trees.extend(copy.deepcopy(evaluated_population))


                preds_train = np.vstack([t.predict(X_train) for t in cumulative_trees])
                preds_test = np.vstack([t.predict(X_test) for t in cumulative_trees])

                ensemble_train_acc = (majority_vote(preds_train) == y_train).mean()
                ensemble_test_acc = (majority_vote(preds_test) == y_test).mean()

                tree_test_accs = [
                    (t.predict(X_test) == y_test).mean()
                    for t in cumulative_trees
                ]

                heights = [t.height() for t in cumulative_trees]
                leaves = [t.num_leaves() for t in cumulative_trees]

                metrics_records.append({
                    "run_id": task_id,
                    "run_num": run_num,
                    "tournament_k": tournament_k,
                    "generation": gen,
                    "avg_tree_test_acc": round(np.mean(tree_test_accs), 3),
                    "ensemble_train_acc": round(ensemble_train_acc, 3),
                    "ensemble_test_acc": round(ensemble_test_acc, 3),
                    "height_var": round(np.var(heights), 3),
                    "leaves_var": round(np.var(leaves), 3),
                    "RF_baseline": round(rf_test_acc, 3),
                    "RF_height": round(rf_depth_var, 3),
                    "RF_leaves": round(rf_leaves_var, 3),
                    "RF_avg": round(rf_avg_score, 3)
                })

                print(
                    f"Gen {gen}: avg_tree={np.mean(tree_test_accs):.4f}, "
                    f"ens_train={ensemble_train_acc:.4f}, "
                    f"ens_test={ensemble_test_acc:.4f}"
                )

            metrics_df = pd.DataFrame(metrics_records)

            metrics_path = os.path.join(
                base_save_folder,
                f"metrics_k{tournament_k}_{task_id}_{run_num}.csv"
            )

            metrics_df.to_csv(metrics_path, index=False)

            print(f"Saved metrics     → {metrics_path}")

    except Exception as e:
        trace = traceback.format_exc()
        print("Failed on ", base_save_folder)
        print(e)
        print(trace)


if __name__ == "__main__":
    main()
    print("DONE")
