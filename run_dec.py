import dataclasses
import time
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from experiments.tree.dec_model import Decoder
from experiments.tree.dec_split_model import DecoderSplit
from experiments.tree.experiment_config import ExperimentConfig, ModelConfig
from experiments.tree.tree_structure import FixSequenceSampleGen, SampleNodesWithoutReplacementGen, DatasetBosEos, \
    SampleLeavesWithReplacementGen, SampleLeavesWithoutReplacementGen, SampleNodesWithReplacementGen, XSampleGen, \
    InterleaveSampleGen

NUM_EPOCHS = 50
SAMPLES_PER_EPOCH = 20_000
RUNNING_BATCH_SIZE = 32
NUM_VAL_TREES = 25
WARMUP_STEPS = 8000

BF = 5; D = 3
base_experiment_config = ExperimentConfig(
    model_config=ModelConfig(
        model='decoder-split',
        n_layer=3,
        n_head=8,
        n_embd=512,
        weight_init=True,
        positional_encoding='global_learn',
        tie_embeddings=False
    ),
    batch_size=64,
    num_runs=3,
    sample_gen=FixSequenceSampleGen(branch_factor=BF, depth=D),
    curriculum=True
)

for experiment in [
    dataclasses.replace(base_experiment_config, sample_gen=FixSequenceSampleGen(branch_factor=BF, depth=D)),
    dataclasses.replace(base_experiment_config, sample_gen=SampleLeavesWithReplacementGen(branch_factor=BF, depth=D)),
    dataclasses.replace(base_experiment_config, sample_gen=SampleNodesWithReplacementGen(branch_factor=BF, depth=D)),
    dataclasses.replace(base_experiment_config, sample_gen=SampleLeavesWithoutReplacementGen(branch_factor=BF, depth=D)),
    dataclasses.replace(base_experiment_config, sample_gen=SampleNodesWithoutReplacementGen(branch_factor=BF, depth=D)),
]:
    if isinstance(experiment, str):
        # Skip commented out experiments
        continue
    for run_id in range(experiment.num_runs):  # Run experiment multiple times

        NUM_LAYERS = experiment.model_config.n_layer
        NUM_HEADS = experiment.model_config.n_head
        N_EMBD = experiment.model_config.n_embd
        MACR_BATCH_SIZE = experiment.batch_size

        BOS_ID = experiment.sample_gen.number_of_nodes  # nodes go from [0, n - 1]
        EOS_ID = BOS_ID + 1
        DEC_BLOCK_SIZE = experiment.sample_gen.number_of_nodes + 1  # decoder sequence length

        if experiment.model == 'decoder':
            model = Decoder(
                vocab_size=experiment.sample_gen.number_of_nodes + 2,
                dec_block_size=DEC_BLOCK_SIZE,
                config=experiment.model_config,
            )
        elif experiment.model == 'decoder-split':
            model = DecoderSplit(
                vocab_size=experiment.sample_gen.number_of_nodes + 2,
                dec_block_size=DEC_BLOCK_SIZE,
                config=experiment.model_config,
            )
        elif experiment.model == 'decoder-split-min':
            model = DecoderSplit(
                vocab_size=experiment.sample_gen.number_of_nodes + 2,
                dec_block_size=DEC_BLOCK_SIZE,
                config=experiment.model_config,
                loss_strategy='min',
                branch_factor=experiment.sample_gen.branch_factor,
            )
        else:
            raise ValueError(f"Unknown model: {experiment.model}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model = torch.compile(model)

        base_lr = 1e-4
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        warmup_steps = WARMUP_STEPS
        d_model = N_EMBD
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)

        if experiment.curriculum:
            print("Curriculum learning")
            max_depth = experiment.sample_gen.depth
            # Run over increasing depths to increase task difficulty incrementally
            sample_gens = [
                dataclasses.replace(experiment.sample_gen, depth=depth) for depth in range(1, max_depth + 1)
            ]
        else:
            print("No curriculum learning")
            sample_gens = [experiment.sample_gen]

        for sample_gen in sample_gens:

            if experiment.curriculum:
                run_name = f"curriculum__{experiment.to_file_name()}__actual__{sample_gen.full_name()}__{run_id}"
            else:
                run_name = f"{experiment.to_file_name()}__{sample_gen.full_name()}__{run_id}"

            print(f"Run {run_name}")
            print("number_of_val_trees", NUM_VAL_TREES)
            print("number_of_nodes", sample_gen.number_of_nodes)
            print("number_of_input_permutations", sample_gen.number_of_input_permutations)
            print("best_possible_loss", sample_gen.best_possible_loss(seq_length=sample_gen.number_of_nodes+1))
            print(f"device: {device}")
            print(f"---")
            print(f"START TRAINING | {run_name}")

            dataset = DatasetBosEos(length=SAMPLES_PER_EPOCH, sample_gen=sample_gen, bos_id=BOS_ID, eos_id=EOS_ID)
            train_loader = DataLoader(dataset, batch_size=RUNNING_BATCH_SIZE, shuffle=True)

            # Sanity check generated tree and tree_correct function to be correct
            for k in range(100):
                dec_x, dec_y = dataset[k]
                if k == 0:
                    print("dec_x", dec_x.tolist())
                    print("dec_y", dec_y.tolist())
                assert sample_gen.tree_correct(dec_x[1:].tolist(), error_to_false=False), f"Tree is generated by dataset, but not correct: {dec_x[1:]}"

            def train(model, data_loader):
                model.train()
                optimizer.zero_grad()
                epoch_start = time.time()
                running_loss_pred = 0.0
                interval_counter = 0
                for xs, ys in data_loader:
                    interval_counter += 1
                    if not isinstance(xs, list):
                        xs = [xs]
                    if not isinstance(ys, list):
                        ys = [ys]
                    xs, ys = [x.to(device) for x in xs], [y.to(device) for y in ys]
                    with torch.autocast(device_type=device):
                        _, loss, loss_pred = model(*xs, *ys)
                    loss.backward()
                    if interval_counter % (batch_size // RUNNING_BATCH_SIZE) == 0:
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        interval_counter = 0

                    running_loss_pred += loss_pred.item()
                epoch_end = time.time()
                train_time = epoch_end - epoch_start
                avg_loss_pred = running_loss_pred / len(train_loader)
                samples_per_sec = (len(train_loader) * train_loader.batch_size) / (epoch_end - epoch_start)
                loss_history.append(avg_loss_pred)
                train_time_history.append(train_time)
                return avg_loss_pred, samples_per_sec, train_time


            def val(model, sample_gen):
                val_start = time.time()
                score = 0.0
                for _ in range(NUM_VAL_TREES):
                    model.eval()
                    with torch.no_grad():
                        pred_tree = model.generate_sample(device=device, bos_id=BOS_ID, eos_id=EOS_ID)
                        pred_tree_correct = sample_gen.tree_correct(pred_tree)
                        score += pred_tree_correct
                        pred_trees[f"epoch_{epoch}"].append((pred_tree, pred_tree_correct))
                val_end = time.time()
                val_time = val_end - val_start
                avg_score = score / max(NUM_VAL_TREES, 1)
                if NUM_VAL_TREES > 0:
                    tree_acc_history.append(avg_score)
                return avg_score, val_time


            loss_history = []
            tree_acc_history = []
            train_time_history = []
            pred_trees = defaultdict(list)
            consecutive_solved_count = 0
            batch_size = MACR_BATCH_SIZE
            for epoch in range(NUM_EPOCHS):

                avg_loss_pred, samples_per_sec, train_time = train(model, train_loader)

                avg_score, val_time = val(model, sample_gen)

                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss (Pred): {avg_loss_pred}, samples/sec: {samples_per_sec:.2f}, sampled tree acc: {avg_score}, val time: {val_time / (train_time + val_time) * 100:.1f}%, lr: {scheduler.get_last_lr()[0]}, bs: {batch_size}")

                if avg_score >= 0.96:
                    consecutive_solved_count += 1
                else:
                    consecutive_solved_count = 0
                if consecutive_solved_count >= 5:
                    print(f"Solved!!")
                    break

            import pickle
            with open(f'/cluster/home/mebr/Master3D/experiments/tree/out/{run_name}.pkl', 'wb') as f:
                pickle.dump({
                    'predicted_trees': pred_trees,
                    'total_samples': SAMPLES_PER_EPOCH * NUM_EPOCHS,
                    'expected_final_loss': sample_gen.best_possible_loss(seq_length=sample_gen.number_of_nodes+1),
                    'loss_history': loss_history,
                    'train_time_history': train_time_history,
                    'tree_acc_history': tree_acc_history,
                }, f)
