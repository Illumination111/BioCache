"""Read-only Query-selectivity diagnostic for trained synthetic checkpoints."""
import argparse
from pathlib import Path
import torch
from biocache.artifacts import write_json
from biocache.data import make_sequence
from run_synthetic import build_model, rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', default='outputs/synthetic_spikingjelly')
    p.add_argument('--output', default='outputs/capacity_diagnostics/query_sensitivity.json')
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    args = p.parse_args()
    torch.set_num_threads(1)
    rows = []
    for seed in args.seeds:
        saved = torch.load(Path(args.checkpoints) / f'lif_{seed}.pt', map_location='cpu', weights_only=False)
        config = saved['config']
        config['device'] = 'cpu'
        model = build_model(config, seed, 'lif')
        model.load_state_dict(saved['model'])
        model.eval()
        for pi, pattern in enumerate(config['data']['patterns']):
            seq = make_sequence(64, 8, 8, seed + 100000 + pi, pattern)
            with torch.no_grad():
                state, _ = rollout(model, seq, 0, 'wait', seed)
                q = (2 * seq.k[:, :, 0, 0])[:, None]
                k, v = seq.k[:, :, 0, 0][:, None], seq.v[:, :, 0, 0][:, None]
                target = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                pred = model.read(state, q)
                centered_pred = pred - pred.mean(dim=2, keepdim=True)
                centered_target = target - target.mean(dim=2, keepdim=True)
                teacher_rms = float(centered_target.square().mean().sqrt())
                student_rms = float(centered_pred.square().mean().sqrt())
                rows.append({'seed': seed, 'pattern': pattern, 'queries_per_history': 8,
                    'mse': float((pred-target).square().mean()),
                    'query_permuted_mse': float((pred.roll(1, dims=2)-target).square().mean()),
                    'mean_value_mse': float((v.mean(dim=2, keepdim=True)-target).square().mean()),
                    'student_query_variation_rms': student_rms,
                    'teacher_query_variation_rms': teacher_rms,
                    'variation_ratio': student_rms / max(teacher_rms, 1e-8)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {
        'protocol': 'Frozen baseline checkpoints; original held-out histories; zero delay; '
                    '8 queries per history, q=2*k_i without noise. Query permutation is cyclic '
                    'over these 8 queries; mean Value reference uses the original history. No retraining.',
        'command': vars(args), 'rows': rows})
    print(output)


if __name__ == '__main__':
    main()
