#%%
from typing import Dict, List, Tuple
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from scipy.stats import entropy
from collections import Counter
import json, os, ot
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def exact_match(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        matches = sum(1 for p, g in zip(preds, gts) if p == g)
        return matches / len(preds) if len(preds) > 0 else 0
    except Exception as e:
        logging.error(f"Error in exact_match: {e}")
        return np.nan

def ngram_overlap(preds: List[List[str]], gts: List[List[str]], n: int = 2) -> float:
    try:
        scores = []
        for p, g in zip(preds, gts):
            p_ng = set(ngrams(p, n))
            g_ng = set(ngrams(g, n))
            if not p_ng or not g_ng:
                scores.append(0)
            else:
                scores.append(len(p_ng & g_ng) / len(p_ng | g_ng))
        return np.mean(scores) if scores else 0
    except Exception as e:
        logging.error(f"Error in ngram_overlap (n={n}): {e}")
        return np.nan

def list_levenshtein(s1: List[str], s2: List[str]) -> int:
    try:
        m, n = len(s1), len(s2)
        if m > n:
            s1, s2 = s2, s1
            m, n = n, m
        current_row = list(range(m + 1))
        for j in range(1, n + 1):
            previous_row, current_row = current_row, [j] + [0] * m
            for i in range(1, m + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                current_row[i] = min(
                    previous_row[i] + 1,   
                    current_row[i-1] + 1,   
                    previous_row[i-1] + cost  
                )
        return current_row[m]
    except Exception as e:
        logging.error(f"Error in list_levenshtein: {e}")
        return np.nan

def list_damerau_levenshtein(s1: List[str], s2: List[str]) -> int:
    try:
        m, n = len(s1), len(s2)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,    
                    dp[i][j-1] + 1,    
                    dp[i-1][j-1] + cost  
                )
                if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-2] + 1)  
        return dp[m][n]
    except Exception as e:
        logging.error(f"Error in list_damerau_levenshtein: {e}")
        return np.nan

def levenshtein(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        pred_flat = [act for trace in preds for act in trace]
        gt_flat = [act for trace in gts for act in trace]
        if not pred_flat and not gt_flat:
            return 0.0
        dist = list_levenshtein(pred_flat, gt_flat)
        if np.isnan(dist):
            return np.nan
        max_len = max(len(pred_flat), len(gt_flat), 1)
        return dist / max_len
    except Exception as e:
        logging.error(f"Error in levenshtein: {e}")
        return np.nan

def damerau_levenshtein(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        pred_flat = [act for trace in preds for act in trace]
        gt_flat = [act for trace in gts for act in trace]
        if not pred_flat and not gt_flat:
            return 0.0
        dist = list_damerau_levenshtein(pred_flat, gt_flat)
        if np.isnan(dist):
            return np.nan
        return dist / max(len(pred_flat), len(gt_flat))
    except Exception as e:
        logging.error(f"Error in damerau_levenshtein: {e}")
        return np.nan

def bleu(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        pred_flat = [act for trace in preds for act in trace]
        gt_flat = [[act for trace in gts for act in trace]]
        if not pred_flat or not gt_flat[0]:
            return 0.0
        smoothing = SmoothingFunction().method4
        return sentence_bleu(gt_flat, pred_flat, smoothing_function=smoothing)
    except Exception as e:
        logging.error(f"Error in bleu: {e}")
        return np.nan

def jaccard_similarity(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        pred_acts = set(act for trace in preds for act in trace)
        gt_acts = set(act for trace in gts for act in trace)
        return len(pred_acts & gt_acts) / len(pred_acts | gt_acts) if pred_acts or gt_acts else 0
    except Exception as e:
        logging.error(f"Error in jaccard_similarity: {e}")
        return np.nan

def avg_trace_length_ratio(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        if not preds or not gts:
            return 0.0
        pred_avg = np.mean([len(trace) for trace in preds])
        gt_avg = np.mean([len(trace) for trace in gts])
        return pred_avg / gt_avg if gt_avg > 0 else 0.0
    except Exception as e:
        logging.error(f"Error in avg_trace_length_ratio: {e}")
        return np.nan

def get_trace_dist(traces: List[List[str]], max_variants: int = 10000) -> Dict[Tuple[str, ...], float]:
    try:
        if not traces:
            return {}
        counter = Counter(tuple(trace) for trace in traces)
        if len(counter) > max_variants:
            logging.warning(f"Too many trace variants ({len(counter)}). Truncating to {max_variants}.")
            counter = dict(counter.most_common(max_variants))
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()} if total > 0 else {}
    except Exception as e:
        logging.error(f"Error in get_trace_dist: {e}")
        return {}

def trace_variant_jsd(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        p_dist = get_trace_dist(preds)
        g_dist = get_trace_dist(gts)
        if not p_dist and not g_dist:
            return 0.0
        if not p_dist:                           
            return 1.0
        if not g_dist:                            
            return 1.0
        all_keys = set(p_dist) | set(g_dist)
        p = np.array([p_dist.get(k, 0) for k in all_keys])
        g = np.array([g_dist.get(k, 0) for k in all_keys])
        epsilon = 1e-12
        p = p + epsilon
        g = g + epsilon
        p /= p.sum()
        g /= g.sum()
        m = 0.5 * (p + g)
        m = 0.5 * (p + g)
        jsd = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(g, m, base=2)
        return float(jsd) 
    except Exception as e:
        logging.error(f"Error in trace_variant_jsd: {e}")
        return 1.0

def trace_variant_tvd(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        p_dist = get_trace_dist(preds)
        g_dist = get_trace_dist(gts)
        if not p_dist and not g_dist:
            return 0.0
        if not p_dist or not g_dist:              
            return 1.0
        all_keys = set(p_dist) | set(g_dist)
        p = np.array([p_dist.get(k, 0.0) for k in all_keys])
        g = np.array([g_dist.get(k, 0.0) for k in all_keys])
        epsilon = 1e-12
        p = p + epsilon
        g = g + epsilon
        p /= p.sum()
        g /= g.sum()
        tvd = 0.5 * np.sum(np.abs(p - g))
        return float(tvd) 
    except Exception as e:
        logging.error(f"Error in trace_variant_tvd: {e}")
        return 1.0

def trace_variant_cosine(preds: List[List[str]], gts: List[List[str]]) -> float:
    try:
        p_dist = get_trace_dist(preds)
        g_dist = get_trace_dist(gts)
        if not p_dist and not g_dist:
            return 1.0
        all_keys = sorted(set(p_dist) | set(g_dist))
        p_vec = np.array([p_dist.get(k, 0) for k in all_keys])
        g_vec = np.array([g_dist.get(k, 0) for k in all_keys])
        if np.all(p_vec == 0) and np.all(g_vec == 0):
            return 1.0
        dot = np.dot(p_vec, g_vec)
        norm_p = np.linalg.norm(p_vec)
        norm_g = np.linalg.norm(g_vec)
        if norm_p == 0 or norm_g == 0:
            return 0.0
        return dot / (norm_p * norm_g)
    except Exception as e:
        logging.error(f"Error in trace_variant_cosine: {e}")
        return np.nan

def trace_variant_emd(
    preds: List[List[str]],
    gts: List[List[str]],
    max_variants: int = 500
) -> float:
    try:
        p_dist = get_trace_dist(preds, max_variants)
        g_dist = get_trace_dist(gts, max_variants)
        if p_dist == g_dist and p_dist:
            return 0.0
        if not p_dist and not g_dist:
            return 0.0
        if not p_dist or not g_dist:
            return float("inf")
        all_keys = sorted(set(p_dist) | set(g_dist))
        p_weights = np.array([p_dist.get(k, 0.0) for k in all_keys], dtype=np.float64)
        g_weights = np.array([g_dist.get(k, 0.0) for k in all_keys], dtype=np.float64)
        p_sum = p_weights.sum()
        g_sum = g_weights.sum()
        if p_sum > 0:
            p_weights /= p_sum
        if g_sum > 0:
            g_weights /= g_sum
        n = len(all_keys)
        if n <= 1:
            return 0.0
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            seq_i = list(all_keys[i])
            for j in range(i + 1, n):
                d = list_levenshtein(seq_i, list(all_keys[j]))
                dist_matrix[i, j] = dist_matrix[j, i] = float(d)
        emd_val = ot.emd2(p_weights, g_weights, dist_matrix, numItermax=200_000)
        return float(emd_val)
    except Exception as e:
        logging.error(f"Error in trace_variant_emd_raw: {e}")
        return float("nan")


def evaluate_predictions(preds: List[List[str]], gts: List[List[str]], config) -> Dict[str, float]:
    results = {
        'predicted_trace_number': len(preds),
        'ground_truth_trace_number': len(gts),
        'exact_match': exact_match(preds, gts),
        'exact_match_p': exact_match(preds, gts[:len(preds)]),
        'levenshtein_p': levenshtein(preds, gts[:len(preds)]),
        'damerau_levenshtein_p': damerau_levenshtein(preds, gts[:len(preds)]),
        'bleu_p': bleu(preds, gts[:len(preds)]),
        'jaccard_similarity_p': jaccard_similarity(preds, gts[:len(preds)]),
        'trace_variant_jsd_p': trace_variant_jsd(preds, gts[:len(preds)]),
        'trace_variant_tvd_p': trace_variant_tvd(preds, gts[:len(preds)]),
        'trace_variant_emd_p': trace_variant_emd(preds, gts[:len(preds)]),
        'avg_trace_length_ratio_p': avg_trace_length_ratio(preds, gts[:len(preds)]),
    }
    
    for n_val in config.get('evaluation', {}).get('ngram_n', []):
        results[f'ngram_overlap_{n_val}'] = ngram_overlap(preds, gts, n=n_val)
    
    return results

def load_logs(file_path: str) -> List[List[str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            traces = [json.loads(line) for line in f if line.strip()]
        return traces
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return []

def split_traces(traces: List[List[str]], config) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    try:
        total = len(traces)
        if total == 0:
            return [], [], []
        train_ratio, val_ratio = config['data']['split_ratios'][:2]
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        return traces[:train_end], traces[train_end:val_end], traces[val_end:]
    except Exception as e:
        logging.error(f"Error in split_traces: {e}")
        return [], [], []

def evaluate_and_save(pred_file: str, test_traces: List[List[str]], config) -> None:
    try:
        pred_traces = load_logs(pred_file)
        results = evaluate_predictions(pred_traces, test_traces, config)
        output_file = os.path.join(config['data']['eval_dir'], os.path.basename(pred_file).replace('.jsonl', '.json'))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved evaluation results to {output_file}")
    except Exception as e:
        logging.error(f"Failed to evaluate {pred_file}: {e}")

def main():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config.json: {e}")
        return
    os.makedirs(config['data']['eval_dir'], exist_ok=True)
    EVAL_DIRECTORY = config['data']['real_dir']
    input_files = [f for f in os.listdir(EVAL_DIRECTORY) if f.endswith('.jsonl')]
    if not input_files:
        logging.warning(f"No input JSONL files found in {EVAL_DIRECTORY}")
        return
    for input_file in sorted(input_files, reverse=True):
        input_path = os.path.join(EVAL_DIRECTORY, input_file)
        traces = load_logs(input_path)
        if not traces:
            continue
        train_traces, val_traces, test_traces = split_traces(traces, config)
        if not test_traces:
            logging.warning(f"No test traces for {input_file}. Skipping.")
            continue
        input_base = input_file[:-6]
        pred_files = [f for f in os.listdir(config['data']['pred_dir']) if f.startswith(input_base + '_') and f.endswith('.jsonl')]
        if not pred_files:
            logging.warning(f"No prediction files found for {input_file}. Skipping.")
            continue
        for pred_file in pred_files:
            eval_path = os.path.join(config['data']['eval_dir'], pred_file[:-1])
            if not os.path.isfile(eval_path):
                pred_path = os.path.join(config['data']['pred_dir'], pred_file)
                evaluate_and_save(pred_path, test_traces, config)
            else:
                print(eval_path, 'Done.')

if __name__ == '__main__':
    main()
#%%