"""Exploratory width/latent-slot controls using the existing training protocol."""
import argparse
from copy import deepcopy
from pathlib import Path
import csv
import json
import statistics
import torch
from biocache.artifacts import start_run, write_json
from biocache.config import load_config
from run_synthetic import train_one, summarize


def average(values):
    return statistics.mean(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/lif_kv_feasibility/configs/synthetic_memory.yaml')
    parser.add_argument('--baseline', default='experiments/lif_kv_feasibility/results/synthetic')
    parser.add_argument('--output', default='outputs/capacity_diagnostics')
    args = parser.parse_args()
    config = load_config(args.config)
    baseline = Path(args.baseline)
    reference_config = load_config(baseline / 'config.yaml')
    for section in ('array', 'readout', 'data', 'training', 'evaluation', 'seeds'):
        if config[section] != reference_config[section]:
            raise ValueError(f'baseline configuration mismatch: {section}')
    if json.loads((baseline / 'run.json').read_text()).get('lif_operator') != 'ParametricLIFNode':
        raise ValueError('baseline must use the current SpikingJelly backend')
    torch.set_num_threads(config['threads'])
    base_n = config['array']['neurons_per_column']
    base_slots = config['readout']['latent_slots']
    cases = [(base_n, base_slots, True), (32, base_slots, False),
             (128, base_slots, False), (base_n, 8, False)]
    root_config = deepcopy(config)
    root_config['diagnostic_cases'] = [dict(neurons=n, latent_slots=m, reuse_reference=reuse)
                                       for n, m, reuse in cases]
    root_config['interpretation'] = ('Exploratory controls, three seeds, fixed 100-step training. '
        'Width also changes W/encoder/reader parameters and compute; it is not an isolated '
        'test of state capacity. No threshold retuning or multiple-comparison correction.')
    output = start_run(args.output, root_config)
    table, detailed = [], []
    for n, slots, reuse in cases:
        case_name = f'N{n}_M{slots}'
        case_config = deepcopy(config)
        case_config['variants'] = ['lif']
        case_config['array']['neurons_per_column'] = n
        case_config['readout']['latent_slots'] = slots
        case_dir = start_run(output / case_name, case_config)
        if reuse:
            results = [json.loads((baseline / f'lif_{seed}.json').read_text()) for seed in config['seeds']]
            for result in results:
                write_json(case_dir / f'lif_{result["seed"]}.json', result)
        else:
            results = [train_one(case_config, case_dir, seed, 'lif') for seed in config['seeds']]
        summary = summarize(results, case_config)
        for row in summary:
            detailed.append({'case': case_name, **row})
        quality = {}
        scopes = {
            'short_last_mse': lambda m: m['view'] == 'last_column' and m['delay'] in (0, 16),
            'wait256_last_mse': lambda m: m['view'] == 'last_column' and m['delay_mode'] == 'wait' and m['delay'] == 256,
            'interference256_last_mse': lambda m: m['view'] == 'last_column' and m['delay_mode'] == 'interference' and m['delay'] == 256,
            'zero_delay_all_mse': lambda m: m['view'] == 'all_columns' and m['delay_mode'] == 'wait' and m['delay'] == 0,
        }
        for name, select in scopes.items():
            samples = [average([m['mse_remote'] for m in r['metrics'] if select(m)]) for r in results]
            quality[name] = average(samples)
            quality[name + '_seed_std'] = statistics.stdev(samples)
        valid_checks = [m for m in summary if m['delay'] > 0]
        entry = {'case': case_name, 'reused_reference': reuse, 'neurons_per_column': n,
                 'total_neurons': n * config['array']['columns'], 'latent_slots': slots,
                 'state_bytes_per_sample': results[0]['metrics'][0]['state_bytes_batch'] // config['data']['test_size'],
                 'parameters': results[0]['parameters']['total'],
                 'synapses': results[0]['parameters']['synapses'],
                 'passed_conditions': sum(m['passes_20pct_vs_both'] for m in valid_checks),
                 'total_conditions': len(valid_checks), **quality}
        table.append(entry)
        write_json(output / 'comparison.json', table)
        write_json(output / 'condition_metrics.json', detailed)
        print(entry, flush=True)
    with (output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    print(f'Capacity diagnostics: {output}', flush=True)


if __name__ == '__main__':
    main()
