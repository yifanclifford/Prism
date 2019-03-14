import pytrec_eval
import scipy.stats


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)


def significant(res1, res2, measure):
    query_ids = list(set(res1.keys()) & set(res2.keys()))
    score1 = [res1[query_id][measure] for query_id in query_ids]
    score2 = [res2[query_id][measure] for query_id in query_ids]
    return scipy.stats.ttest_rel(score1, score2)
