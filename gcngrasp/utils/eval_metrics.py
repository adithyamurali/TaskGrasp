from collections import defaultdict
import os
from pprint import pprint

import numpy as np
from sklearn.metrics import average_precision_score

from utils.splits import get_ot_pairs_taskgrasp

class APMetrics:
    def __init__(self, cfg):
        self.cfg = cfg
        task1_results_file = os.path.join(
            self.cfg.base_dir, self.cfg.folder_dir, 'task1_results.txt')
        assert os.path.exists(task1_results_file)

        if self.cfg.dataset_class in ['SGNTaskGrasp', 'GCNTaskGrasp', 'BaselineData']:
            object_task_pairs = get_ot_pairs_taskgrasp(task1_results_file)
            self.TASK2_ot_pairs = object_task_pairs['True'] + \
                object_task_pairs['Weak True']
        else:
            raise ValueError('Unknown dataset class {}'.format(self.cfg.dataset_class))

    """
    results: a list of dicts with keys "labels", "probs", <key>
    key: the key to aggregate over; e.g. if it is "task_ids", we would compute the mAP over tasks
    """
    def _mean_ap(self, results, key):
        aggregate_by_key = defaultdict(lambda: {"labels": [], "probs": []})
        for x in results:
            aggregate_by_key[x[key]]["labels"].append(x["labels"])
            aggregate_by_key[x[key]]["probs"].append(x["probs"])
        aps = []
        failed_cnt = 0
        for el_key in aggregate_by_key:
            labels = np.array(aggregate_by_key[el_key]["labels"])
            probs = np.array(aggregate_by_key[el_key]["probs"])
            ap = average_precision_score(labels, probs)
            if np.isnan(ap):
                failed_cnt += 1
            else:
                aps.append(ap)
        print(f"{key} failed: {failed_cnt}")
        mean_ap = sum(aps)/len(aps)
        return mean_ap
    
    """
    outputs: all results as a list of dicts; required keys for the dict:
        "labels", "probs", "task_ids", "instance_ids", "class_ids"

    computes mAP over instance, class, and task; both filtered and not filtered for valid task-object pairs

    For the filtering to work, instance_ids have to be of the form in the task1 results file (e.g. 128_XXX) and
    tasks need to be the word of the task, not just a numerical id
    """
    def get_map(self, results):
        metrics = {}
        
        metrics["unfiltered instance_mAP"] = self._mean_ap(results, "instance_ids")
        metrics["unfiltered class_mAP"] = self._mean_ap(results, "class_ids")
        metrics["unfiltered task_mAP"] = self._mean_ap(results, "task_ids")

        results.sort(key=lambda x: x["instance_ids"])

        # filtering for valid task-object pairs
        print("len results", len(results))
        results = [x for x in results
                    if f'{x["instance_ids"]}-{x["task_ids"]}'
                    in self.TASK2_ot_pairs]
        print("len results filtered", len(results))

        metrics["instance_mAP"] = self._mean_ap(results, "instance_ids")
        metrics["class_mAP"] = self._mean_ap(results, "class_ids")
        metrics["task_mAP"] = self._mean_ap(results, "task_ids")
        
        return metrics