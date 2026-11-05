"""One fixed shared recovery SFT under the persistent holder lease."""
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from src.agents import order_recovery_acceptance as acceptance
from src.agents import order_recovery_sft_validation as sft_validation
from src.agents import order_recovery_sft_worker as sft_worker
from src.agents.coldstart_sft import read_examples
from src.agents.gpu_memory_coordinator import (
    GIB,
    PHASES,
    config_for_holder,
    controller_idle,
    freeze_config,
)
from src.agents.order_recovery_controller import release_for_inference
from src.agents.order_recovery_gpu_monitor import run_monitored
from src.agents.order_recovery_probe import model_contracts, validate_model

ROOT = Path('/home/tianming/projects/finance-credit-agent')
RUN = ROOT / 'runs/m2-order-recovery-sft-20261011-v1'
BASE = ROOT / 'runs/m2-coldstart-v3-20260914/merged_model'
DATA = RUN / 'train.jsonl'


def save(name, payload):
    p = RUN / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    sft_validation.refuse_execution_overwrite(RUN)
    state = {'status': 'PREPARING', 'automatic_retry': False, 'base': str(BASE)}
    save('controller_state.json', state)
    try:
        from transformers import AutoTokenizer

        validate_model(BASE, model_contracts(ROOT / 'runs/m2-parameter-grid-20261008-r5')['sft'])
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        examples = read_examples(DATA, tokenizer, 8192)
        if len(examples) != 384:
            raise ValueError('The frozen SFT budget requires exactly 384 examples')
        dataset_manifest_path = RUN / 'dataset_manifest.json'
        bank_path = RUN / 'acceptance_bank.json'
        dataset = json.loads(dataset_manifest_path.read_text())
        bank = json.loads(bank_path.read_text())
        sft_validation.validate_dataset_receipt(RUN, dataset, bank)
        if (dataset['status'] != 'COMPLETE' or dataset['rows'] != 384
                or dataset['recovery_rows'] != 192 or dataset['unchanged_replay_rows'] != 192
                or dataset['acceptance_is_training'] is not False
                or dataset['training_observation_profile'] != 'history_features_recovery_v2'
                or dataset['train_jsonl_sha256'] != digest(DATA)
                or dataset['acceptance_bank_sha256'] != digest(bank_path)
                or bank['training_dataset_sha256'] != digest(DATA)
                or len(bank['cases']) != 24
                or not dataset['training_acceptance_known_failure_dates_disjoint']):
            raise ValueError('Frozen shared SFT data or independent acceptance bank differs')
        sources = acceptance.runtime_sources()
        sources[str(Path(__file__).resolve())] = digest(Path(__file__))
        helper = Path(sft_validation.__file__).resolve()
        sources[str(helper)] = digest(helper)
        with zipfile.ZipFile(RUN / 'runtime_sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for filename, expected in sources.items():
                if digest(Path(filename)) != expected:
                    raise ValueError(f'SFT source changed before snapshot: {filename}')
                archive.write(filename, arcname=filename.lstrip('/'))
        settings = {
            'model': str(BASE), 'data': str(DATA), 'data_sha256': digest(DATA),
            'controller_sha256': digest(Path(__file__)), 'seed': 42,
            'epochs': 3, 'micro_batch': 1, 'gradient_accumulation': 16,
            'learning_rate': 5e-5, 'max_length': 8192, 'optimizer_steps': 72,
            'completion_only': True, 'truncate': False,
            'normal_replay': 192, 'new_recovery_demonstrations': 192,
            'observation_profile': 'history_features_recovery_v2',
            'acceptance_used_for_training': False,
            'dataset_manifest_sha256': digest(dataset_manifest_path),
            'acceptance_bank_sha256': digest(bank_path),
            'dataset_source_receipt_sha256': digest(RUN / 'dataset_source_receipt.json'),
            'runtime_sha256': sources,
            'runtime_snapshot_sha256': digest(RUN / 'runtime_sources.zip'),
            'base_merge_receipt_sha256': digest(BASE / 'merge_receipt.json'),
            'acceptance_gate': acceptance.ENDPOINT_GATE,
            'acceptance_gate_sha256': acceptance.fingerprint(acceptance.ENDPOINT_GATE),
            'plan_sha256': digest(ROOT / 'docs/2026-10-11_M2_ORDER_RECOVERY_LEARNING_PLAN.md'),
            'training_allocator': {
                'configuration': sft_worker.ALLOCATOR_CONFIG,
                'process_allocator_cap_bytes': sft_worker.ALLOCATOR_CAP_BYTES,
                'aggregate_stop_bytes': 80 * GIB,
            },
        }
        if dataset['suggested_fixed_training_budget'] != {
            'epochs': 3, 'micro_batch': 1, 'gradient_accumulation': 16,
            'optimizer_updates': 72, 'learning_rate': 5e-5,
        }:
            raise ValueError('Predeclared dataset training budget differs')
        save('training_contract.json', settings)
        config = config_for_holder(
            ROOT / 'runs/gpu-reservation-persistent-20261007',
            lease_token='shared-recovery-sft-20261011-v1',
            evidence_directory=RUN / 'gpu_handoff',
            phase_peak_bytes={phase: 60 * GIB for phase in PHASES},
            wait_timeout_seconds=120,
        )
        freeze_config(RUN / 'gpu_handoff/config.json', config)
        admission = release_for_inference(config)
        baseline = admission['driver_total_bytes'] - admission['driver_free_bytes']
        if admission['driver_free_bytes'] < 60 * GIB or baseline > 16 * GIB:
            raise RuntimeError('Shared SFT requires 60 GiB free and at most 16 GiB baseline use')
        state.update(status='TRAINING', admission=admission,
                     aggregate_baseline_bytes=baseline)
        save('controller_state.json', state)
        command = [
            sys.executable, '-u', '-m', 'src.agents.order_recovery_sft_worker',
            '--allocator-receipt', str(RUN / 'training_allocator_receipt.json'),
            '--data', str(DATA), '--model', str(BASE), '--output', str(RUN / 'training'),
            '--seed', '42', '--epochs', '3', '--micro-batch', '1',
            '--gradient-accumulation', '16', '--lr', '5e-5', '--max-length', '8192',
        ]
        state['training_command'] = command
        with (RUN / 'training.log').open('x') as log:
            run_monitored(
                command, config=config, evidence_path=RUN / 'gpu-training.jsonl',
                timeout=3600, check=True, cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
        deadline = time.monotonic() + 30
        while True:
            try:
                controller_idle(config)
                break
            except RuntimeError as error:
                if 'child processes remain' not in str(error) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        training_evidence = sft_validation.validate_training_evidence(
            RUN / 'training', examples, settings,
            implementation=ROOT / 'src/agents/coldstart_sft.py',
            allocator_receipt=RUN / 'training_allocator_receipt.json',
        )
        save('training_evidence_readout.json', training_evidence)
        state['status'] = 'MERGING_ON_CPU'
        save('controller_state.json', state)
        command = [
            sys.executable, '-u', '-m', 'src.agents.coldstart_sft', 'merge',
            '--model', str(BASE), '--adapter', str(RUN / 'training/adapter'),
            '--output', str(RUN / 'merged_model'),
        ]
        state['merge_command'] = command
        with (RUN / 'merge.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, check=True, timeout=1800)
        merge_evidence = sft_validation.validate_merge_evidence(
            RUN / 'merged_model', base=BASE, adapter=RUN / 'training/adapter',
            inventory=acceptance.hf_inventory,
        )
        save('merge_evidence_readout.json', merge_evidence)
        summary = training_evidence['summary']
        if (summary['status'] != 'COMPLETE' or summary['optimizer_steps'] != 72
                or summary['probe_changed_steps'] != 72
                or summary['data_sha256'] != settings['data_sha256']
                or digest(DATA) != settings['data_sha256']
                or digest(dataset_manifest_path) != settings['dataset_manifest_sha256']
                or digest(bank_path) != settings['acceptance_bank_sha256']
                or digest(RUN / 'dataset_source_receipt.json') != (
                    settings['dataset_source_receipt_sha256'])):
            raise ValueError('Shared SFT update evidence or frozen data differs')
        for filename, expected in sources.items():
            if digest(Path(filename)) != expected:
                raise ValueError(f'Frozen SFT runtime changed: {filename}')
        files = acceptance.hf_inventory(RUN / 'merged_model', excluded=set())
        evidence_paths = [RUN / 'training_contract.json', RUN / 'training/summary.json',
                          RUN / 'merged_model/merge_receipt.json', dataset_manifest_path,
                          RUN / 'dataset_source_receipt.json', DATA,
                          ROOT / 'docs/2026-10-11_M2_ORDER_RECOVERY_LEARNING_PLAN.md']
        evidence_paths.extend([
            RUN / 'training/settings.json', RUN / 'training/data_preflight.json',
            RUN / 'training/steps.jsonl', RUN / 'training_allocator_receipt.json',
            RUN / 'training_evidence_readout.json', RUN / 'merge_evidence_readout.json', bank_path,
        ])
        lineage = {
            'status': 'COMPLETE', 'model_key': 'sft', 'role': 'common_sft',
            'model_fingerprint': acceptance.fingerprint(files), 'files': files,
            'training_updates': 72, 'training_seed': 42,
            'action_protocol': 'single_tool_portfolio_v1',
            'observation_profile': 'history_features_recovery_v2',
            'acceptance_gate_sha256': settings['acceptance_gate_sha256'],
            'training_evidence_sha256': {str(p): digest(p) for p in evidence_paths},
            'common_initialization_for_all_later_arms': True,
        }
        save('lineage_receipt.json', lineage)
        save('model_spec_sft.json', {'sft': {
            'model_path': str(RUN / 'merged_model'),
            'lineage_receipt': str(RUN / 'lineage_receipt.json'),
        }})
        state.update(status='COMPLETE', training_summary=summary,
                     merged_model=str(RUN / 'merged_model'),
                     model_fingerprint=lineage['model_fingerprint'])
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save('controller_state.json', state)


if __name__ == '__main__':
    main()
