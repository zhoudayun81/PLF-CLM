#%%
import os
import json
from typing import Dict, List
import torch
import optuna
import transformers
from transformers import BertConfig, GPT2Config, T5Config, BertLMHeadModel, GPT2LMHeadModel, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, BertTokenizer, EarlyStoppingCallback
from datasets import Dataset
from packaging import version
import nltk
import gc
import shutil
import random
from collections import Counter
from measure import evaluate_predictions

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
nltk.download('punkt', quiet=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

def build_vocab(traces: List[List[str]]) -> List[str]:
    activities = sorted(set(act for trace in traces for act in trace))
    tokens = [config['model']['pad_token'], config['model']['bos_token'], config['model']['eos_token']] + activities
    return tokens

def create_tokenizer(vocab_tokens: List[str], vocab_file: str):
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in vocab_tokens:
            f.write(f"{token}\n")
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False, tokenize_chinese_chars=False, unk_token=None)
    tokenizer.add_special_tokens({'pad_token': config['model']['pad_token'], 'bos_token': config['model']['bos_token'], 'eos_token': config['model']['eos_token']})
    return tokenizer

def trace_to_ids(trace: List[str], tokenizer: BertTokenizer) -> List[int]:
    ids = []
    for activity in trace:
        tid = tokenizer.convert_tokens_to_ids(activity)
        ids.append(tid)
    return ids

def concatenate_traces(traces: List[List[str]], tokenizer: BertTokenizer) -> List[int]:
    sequence = [tokenizer.bos_token_id]
    for trace in traces:
        for activity in trace:
            token_id = tokenizer.convert_tokens_to_ids(activity)
            sequence.append(token_id)
        sequence.append(tokenizer.eos_token_id)
    return sequence

def chunk_sequence(seq: List[int], max_pos: int, overlap=0) -> List[List[int]]:
    chunks = []
    step = max_pos - overlap
    for i in range(0, len(seq), step):
        chunk = seq[i:i + max_pos]
        if len(chunk) > 1:  # keep only valid chunks
            chunks.append(chunk)
        if i + max_pos >= len(seq):
            break
    return chunks

def prepare_lm_dataset(chunks: List[List[int]], max_pos: int, pad_id: int) -> Dataset:
    data = []
    for chunk in chunks:
        if len(chunk) > max_pos:
            chunk = chunk[:max_pos]  # truncate
        else:
            chunk = chunk + [pad_id] * (max_pos - len(chunk))  # pad
        attention_mask = [1 if token != pad_id else 0 for token in chunk]
        data.append({'input_ids': chunk, 'labels': chunk, 'attention_mask': attention_mask})
    return Dataset.from_list(data)

def get_hf_model(model_type: str, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int, max_pos: int):
    if model_type == "encoder_only":
        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_pos,
            is_decoder=True, # Required for causal LM
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            type_vocab_size=1,
        )
        model = BertLMHeadModel(cfg)
    elif model_type == "decoder_only":
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_pos,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = GPT2LMHeadModel(cfg)
    elif model_type == "enc_dec":
        cfg = T5Config(
            vocab_size=vocab_size,
            d_model=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            #d_ff=hidden_dim * 4,
            decoder_start_token_id=2,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = T5ForConditionalGeneration(cfg)
    else:
        model = None
        print(f"Unknown model type: {model_type}")
    return model

def objective(trial, model_type, train_dataset, val_dataset, tokenizer, config):
    def get_valid_num_heads(hidden_dim: int) -> List[int]:
        divisors = []
        for i in range(1, hidden_dim + 1):
            if hidden_dim % i == 0 and i <= 16:
                divisors.append(i)
        return divisors if divisors else [1]
    lr = trial.suggest_float('lr', config['model']['lr'][0], config['model']['lr'][-1], log=True)
    batch_size = trial.suggest_categorical('batch_size', config['model']['batch_size'])
    num_layers = trial.suggest_int('num_layers', config['model']['num_layers'][0], config['model']['num_layers'][-1])
    hidden_dim = trial.suggest_categorical('hidden_dim', config['model']['hidden_dim'])
    valid_num_heads = get_valid_num_heads(hidden_dim)
    num_heads = trial.suggest_categorical('num_heads', valid_num_heads)
    num_epochs = trial.suggest_int('num_epochs', config['training']['epochs'][0], config['training']['epochs'][-1]) 
    model = get_hf_model(model_type, tokenizer.vocab_size, hidden_dim, num_layers, num_heads, config['model']['max_pos'])
    model.gradient_checkpointing_enable()
    transformers_version = version.parse(transformers.__version__)
    trial_dir = f'./temp/trial_{trial.number}'
    training_args_kwargs = {
        'output_dir': trial_dir,
        'num_train_epochs': num_epochs,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': lr,
        'seed': config['training']['seed'],
        'save_strategy': 'epoch',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        #'bf16': True,
    }
    if transformers_version >= version.parse('4.28.0'):
        training_args_kwargs['eval_strategy'] = 'epoch'
    else:
        training_args_kwargs['evaluation_strategy'] = 'epoch'
    training_args = TrainingArguments(**training_args_kwargs)
    if model_type == "enc_dec":
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['training']['early_stop_patience'])],
    )
    trainer.train()
    eval_results = trainer.evaluate()
    del trainer, model  # Explicitly delete
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()  # Clear GPU memory
    shutil.rmtree(trial_dir, ignore_errors=True) 
    return eval_results['eval_loss']

def final_train(model_type, best_params, train_dataset, val_dataset, tokenizer, config, device, filename):
    model = get_hf_model(
        model_type, 
        tokenizer.vocab_size, 
        best_params['hidden_dim'], 
        best_params['num_layers'], 
        best_params['num_heads'], 
        config['model']['max_pos']
    ).to(device)
    transformers_version = version.parse(transformers.__version__)
    final_dir = f'./temp/final_{filename[:-6]}_{model_type}'
    training_args_kwargs = {
        'output_dir': final_dir, 
        'num_train_epochs': best_params['num_epochs'],
        'per_device_train_batch_size': best_params['batch_size'],
        'per_device_eval_batch_size': best_params['batch_size'],
        'learning_rate': best_params['lr'],
        'seed': config['training']['seed'],
        'save_strategy': 'epoch',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        #'bf16': True,
    }
    if transformers_version >= version.parse('4.28.0'):
        training_args_kwargs['eval_strategy'] = 'epoch'
    else:
        training_args_kwargs['evaluation_strategy'] = 'epoch'
    training_args = TrainingArguments(**training_args_kwargs)
    if model_type == "enc_dec":
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # Causal LM
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['training']['early_stop_patience'])],
    )
    trainer.train()
    del trainer  # Explicitly delete
    gc.collect()  # Force garbage collection
    torch.cuda.empty_cache()  # Clear GPU memory
    if config['model']['save']:
        model_path = os.path.join(config['data']['model_dir'], f"{filename[:-6]}_{model_type}.pt")
        torch.save(model, model_path)
    shutil.rmtree(final_dir, ignore_errors=True)
    return model

def generate_for_window(model, model_type, prefix_ids, tokenizer, device, temperature=1.0, max_new_tokens=512):
    model.eval()
    unk_id = tokenizer.unk_token_id 
    max_pos = model.config.max_position_embeddings if model_type == "encoder_only" else model.config.n_positions if model_type == "decoder_only" else config['model']['max_pos']
    if len(prefix_ids) > max_pos:
        prefix_ids = prefix_ids[-max_pos:] 
    effective_max_new = min(max_new_tokens, max_pos - len(prefix_ids))
    if effective_max_new <= 0:
        return [] 
    prefix = torch.tensor([prefix_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(prefix).to(device)
    if tokenizer.pad_token_id in prefix:
        attention_mask[prefix == tokenizer.pad_token_id] = 0
    with torch.no_grad():
        if model_type == "decoder_only":
            generated = model.generate(
                input_ids=prefix,
                attention_mask=attention_mask,
                max_new_tokens=effective_max_new, 
                temperature=temperature,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[unk_id]],
            )
            new_ids = generated[0, len(prefix_ids):].cpu().tolist()
        elif model_type == "enc_dec":
            generated = model.generate(
                input_ids=prefix,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                decoder_start_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[unk_id]],
            )
            new_ids = generated[0].cpu().tolist()
        elif model_type == "encoder_only":
            generated = prefix.clone()
            safe_max_new = max(0, max_pos - generated.shape[1])
            max_new_tokens = min(max_new_tokens, safe_max_new)
            for _ in range(max_new_tokens):
                if generated.shape[1] >= max_pos:
                    break
                outputs = model(input_ids=generated, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                logits[:, unk_id] = -float('inf')
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones((generated.shape[0], 1), device=device)), dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break 
            new_ids = generated[0, len(prefix_ids):].cpu().tolist()
    return new_ids

def extract_n_traces(gen_ids, n, tokenizer):
    traces, cur = [], []
    for tid in gen_ids:
        tok = tokenizer._convert_id_to_token(tid)
        if tok in [tokenizer.bos_token, tokenizer.pad_token]:
            continue
        elif tok == tokenizer.eos_token:
            if cur:
                traces.append(cur)
                cur = []
                if len(traces) == n:
                    break
        else:
            cur.append(tok)
    if cur:
        traces.append(cur)
    traces = [t for t in traces if len(t) > 0]
    if len(traces) > n:
        traces = traces[:n]
    return traces

def recursive_no_overlap_generate(model, model_type, tokenizer, device, prefix_traces: List[List[str]], test_traces: List[List[str]], config) -> List[List[str]]:
    if not test_traces:
        return []
    produced = [] 
    while len(produced) < len(test_traces):
        prefix_ids = [tokenizer.bos_token_id]
        for trace in prefix_traces:
            ids = trace_to_ids(trace, tokenizer)
            prefix_ids.extend(ids)
            prefix_ids.append(tokenizer.eos_token_id)
        est_target_block = test_traces[len(produced): len(produced) + config['data']['pred_horizon']]
        est_length = max(1, sum(len(t) for t in est_target_block))
        max_new = max(4, int(est_length * 1.2))
        gen_ids = generate_for_window(model, model_type, prefix_ids, tokenizer, device, max_new_tokens=max_new)
        torch.cuda.empty_cache()  # Clear GPU memory
        new_traces = extract_n_traces(gen_ids, config['data']['pred_horizon'], tokenizer)
        produced.extend(new_traces)
        combined = prefix_traces + new_traces
        prefix_traces = combined[-config['data']['window_size']:]
        if all((not t) for t in new_traces):
            break
    return produced[:len(test_traces)]

def process_file(traces: List[List[str]], filename: str, config: Dict, device):
    vocab = build_vocab(traces)
    vocab_file = os.path.join(config['data']['model_dir'], f"{filename[:-6]}_vocab.txt")
    tokenizer = create_tokenizer(vocab, vocab_file)
    sub_dir = os.path.join(config['data']['model_dir'], filename[:-6])
    tokenizer.save_pretrained(sub_dir)
    total = len(traces)
    train_end = int(total * config['data']['split_ratios'][0])
    val_end = train_end + int(total * config['data']['split_ratios'][1])
    train_traces = traces[:train_end]
    val_traces = traces[train_end:val_end]
    test_traces = traces[val_end:]
    train_seq = concatenate_traces(train_traces, tokenizer)
    val_seq = concatenate_traces(val_traces, tokenizer)
    train_chunks = chunk_sequence(train_seq, config['model']['max_pos'], int(config['model']['max_pos']*0.1))
    val_chunks = chunk_sequence(val_seq, config['model']['max_pos'], int(config['model']['max_pos']*0.1))
    train_dataset = prepare_lm_dataset(train_chunks, max_pos=config['model']['max_pos'], pad_id=tokenizer.pad_token_id)
    val_dataset = prepare_lm_dataset(val_chunks, max_pos=config['model']['max_pos'], pad_id=tokenizer.pad_token_id)
    trainval = train_traces + val_traces
    if len(trainval) < config['data']['window_size']:
        prefix_traces = trainval
    else:
        prefix_traces = trainval[-config['data']['window_size']:]
    evals_all = {}
    for model_type in config['model']['types']:
        shutil.rmtree('./temp', ignore_errors=True)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, model_type, train_dataset, val_dataset, tokenizer, config), n_trials=config['training']['optuna_trials'])
        best_params = study.best_params
        full_seq = concatenate_traces(train_traces + val_traces, tokenizer)
        full_chunks = chunk_sequence(full_seq, config['model']['max_pos'], int(config['model']['max_pos']*0.1))
        full_dataset = prepare_lm_dataset(full_chunks, max_pos=config['model']['max_pos'], pad_id=tokenizer.pad_token_id)
        model = final_train(model_type, best_params, full_dataset, val_dataset, tokenizer, config, device, filename)
        model.to(device)
        preds = recursive_no_overlap_generate(model, model_type, tokenizer, device, prefix_traces.copy(), test_traces, config)
        pred_path = os.path.join(config['data']['pred_dir'], f"{filename[:-6]}_{model_type}.jsonl")
        with open(pred_path, 'w', encoding='utf-8') as f:
            for trace in preds:
                json.dump(trace, f)
                f.write('\n')
        eval_results = evaluate_predictions(preds, test_traces, config)
        eval_path = os.path.join(config['data']['eval_dir'], f"{filename[:-6]}_{model_type}.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=4)
        evals_all[model_type] = eval_results

    return evals_all

def generate_from_saved_model(traces: List[List[str]], filename: str, config: Dict, device):
    total = len(traces)
    train_end = int(total * config['data']['split_ratios'][0])
    val_end = train_end + int(total * config['data']['split_ratios'][1])
    train_traces = traces[:train_end]
    val_traces = traces[train_end:val_end]
    test_traces = traces[val_end:]
    sub_dir = os.path.join(config['data']['model_dir'], filename[:-6])
    tokenizer = BertTokenizer.from_pretrained(sub_dir, do_lower_case=False, tokenize_chinese_chars=False, unk_token=None)
    trainval = train_traces + val_traces
    if len(trainval) < config['data']['window_size']:
        prefix_traces = trainval
    else:
        prefix_traces = trainval[-config['data']['window_size']:]
    evals_all = {}
    for model_type in config['model']['types']:
        model_path = os.path.join(sub_dir, f"{model_type}.pt")
        if not os.path.exists(model_path):
            print(f"Skipping {model_type} for {filename}: Saved model {model_path} not found.")
            continue
        model = torch.load(model_path, map_location=device)
        model.to(device)
        preds = recursive_no_overlap_generate(model, model_type, tokenizer, device, prefix_traces.copy(), test_traces, config)
        pred_path = os.path.join(config['data']['pred_dir'], f"{filename[:-6]}_{model_type}.jsonl")
        with open(pred_path, 'w', encoding='utf-8') as f:
            for trace in preds:
                json.dump(trace, f)
                f.write('\n')
        eval_results = evaluate_predictions(preds, test_traces, config)
        eval_path = os.path.join(config['data']['eval_dir'], f"{filename[:-6]}_{model_type}.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=4)
        evals_all[model_type] = eval_results
    return evals_all

def most_frequent_baseline(traces: List[List[str]], test_len: int) -> List[List[str]]:
    most_freq = Counter(tuple(trace) for trace in traces).most_common(1)[0][0]
    return [list(most_freq) for _ in range(test_len)]

def random_sampling_baseline(traces: List[List[str]], test_len: int) -> List[List[str]]:
    trace_tuples = set(tuple(trace) for trace in traces)  # Unique traces
    unique_traces = list(trace_tuples)
    sampled_tuples = random.choices(unique_traces, k=test_len)
    return [list(t) for t in sampled_tuples]

def weighted_random_sampling_baseline(traces: List[List[str]], test_len: int) -> List[List[str]]:
    trace_tuples = [tuple(trace) for trace in traces]  # Use tuples for hashing
    freq = Counter(trace_tuples)
    unique_traces = list(freq.keys())
    weights = [freq[t] for t in unique_traces]
    sampled_tuples = random.choices(unique_traces, weights=weights, k=test_len)
    return [list(t) for t in sampled_tuples]  # Convert back to lists

def process_file_baseline(traces: List[List[str]], filename: str, config: Dict, device):
    total = len(traces)
    train_end = int(total * config['data']['split_ratios'][0])
    val_end = train_end + int(total * config['data']['split_ratios'][1])
    train_traces = traces[:train_end]
    val_traces = traces[train_end:val_end]
    test_traces = traces[val_end:]
    train_val = train_traces + val_traces
    evals_all = {}
    
    preds = most_frequent_baseline(train_val, len(test_traces))
    pred_path = os.path.join(config['data']['pred_dir'], f"{filename[:-6]}_most_frequent.jsonl")
    with open(pred_path, 'w', encoding='utf-8') as f:
        for trace in preds:
            json.dump(trace, f)
            f.write('\n')
    eval_results = evaluate_predictions(preds, test_traces, config)
    eval_path = os.path.join(config['data']['eval_dir'], f"{filename[:-6]}_most_frequent.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4)
    evals_all['most_frequent'] = eval_results

    preds = random_sampling_baseline(train_val, len(test_traces))
    pred_path = os.path.join(config['data']['pred_dir'], f"{filename[:-6]}_random_sampling.jsonl")
    with open(pred_path, 'w', encoding='utf-8') as f:
        for trace in preds:
            json.dump(trace, f)
            f.write('\n')
    eval_results = evaluate_predictions(preds, test_traces, config)
    eval_path = os.path.join(config['data']['eval_dir'], f"{filename[:-6]}_random_sampling.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4)
    evals_all['random_sampling'] = eval_results
    
    preds = weighted_random_sampling_baseline(train_val, len(test_traces))
    pred_path = os.path.join(config['data']['pred_dir'], f"{filename[:-6]}_weighted_random_sampling.jsonl")
    with open(pred_path, 'w', encoding='utf-8') as f:
        for trace in preds:
            json.dump(trace, f)
            f.write('\n')
    eval_results = evaluate_predictions(preds, test_traces, config)
    eval_path = os.path.join(config['data']['eval_dir'], f"{filename[:-6]}_weighted_random_sampling.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=4)
    evals_all['weighted_random_sampling'] = eval_results
    return evals_all

def load_logs(file_path: str) -> List[List[str]]:
    try:
        with open(file_path, 'r') as f:
            traces = [json.loads(line) for line in f]
        return traces
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}. Skipping.")
        return None

if __name__ == '__main__':
    os.makedirs(config['data']['model_dir'], exist_ok=True)
    os.makedirs(config['data']['pred_dir'], exist_ok=True)
    os.makedirs(config['data']['eval_dir'], exist_ok=True)

    real_dir = config['data']['real_dir']
    for filename in os.listdir(real_dir):
        if not filename.endswith('.jsonl'):
            continue
        file_path = os.path.join(real_dir, filename)
        print(f'Attempting to process file: {filename}')
        traces = load_logs(file_path)
        if traces is None:
            print(f'Skipping {filename}: File not found or could not be loaded.')
            continue
        if not traces or all(not trace for trace in traces):
            print(f'Skipping {filename}: Traces are empty or contain only empty lists.')
            continue
        print(f'Training File: {filename}')
        evals = process_file(traces, filename, config, device)

# %%
