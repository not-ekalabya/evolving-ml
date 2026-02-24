import numpy as np
import torch
import torch.nn as nn

import main


class PolicyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def graph_features(model, last_acc, last_reward):
    num_nodes = len(model.execution_order)
    num_matmul = sum(1 for n in model.execution_order if isinstance(model.nodes[n], main.MatMulNode))
    params = main.count_parameters(model)
    depth = num_nodes
    return torch.tensor(
        [
            float(num_nodes),
            float(num_matmul),
            float(params),
            float(depth),
            float(last_acc),
            float(last_reward),
        ],
        dtype=torch.float32,
    )


def run_single_graph_rl(
    iterations=200,
    training_steps=100,
    val_batches=50,
    lambda_size=1e-4,
    entropy_coef=0.01,
    batch_size=64,
    random_action_prob=0.3,
    novelty_weight=0.05,
    patience=20,
    seed=42,
):
    device = main.get_device(strict=True)
    rng = np.random.RandomState(seed)
    train_loader, test_loader = main.get_dataloaders(batch_size=batch_size)

    actions = ["insert_layer", "change_width", "change_activation", "add_skip", "add_concat"]
    policy_device = "cpu"
    policy = PolicyNet(input_dim=6, hidden_dim=32, output_dim=len(actions)).to(policy_device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    model = main.build_minimal_model().to(device)
    _ = main.train_brief(model, train_loader, device, steps=training_steps)
    last_acc = main.evaluate_accuracy(model, test_loader, device, max_batches=val_batches)
    last_reward = 0.0
    no_improve = 0
    best_acc = last_acc

    for step in range(iterations):
        state = graph_features(model, last_acc, last_reward).to(policy_device)
        logits = policy(state)
        log_probs = torch.log_softmax(logits, dim=0)
        probs = torch.softmax(logits, dim=0)

        available = main.available_mutations(model)
        if not available:
            available = ["insert_layer"]

        if rng.rand() < random_action_prob:
            action = rng.choice(available)
            action_idx = torch.tensor(actions.index(action))
            masked_probs = torch.tensor([1.0 if a == action else 0.0 for a in actions], dtype=probs.dtype)
        else:
            mask = torch.tensor([1.0 if a in available else 0.0 for a in actions], dtype=probs.dtype)
            masked_probs = probs * mask
            if masked_probs.sum().item() <= 0:
                masked_probs = mask
            masked_probs = masked_probs / masked_probs.sum()
            action_idx = torch.multinomial(masked_probs, 1).squeeze(0)
            action = actions[int(action_idx.item())]

        candidate = None
        ok = False
        candidate_is_valid = False
        for _ in range(5):
            candidate, ok = main.apply_mutation_action(model, rng, action)
            if ok:
                candidate.to(device)
                candidate_is_valid = main.is_valid_model(candidate, device)
                if candidate_is_valid:
                    break
            candidate = None
            ok = False
            candidate_is_valid = False

        if not ok or not candidate_is_valid:
            reward = -0.1
            accept = False
        else:
            trained = main.train_brief(candidate, train_loader, device, steps=training_steps)
            if not trained:
                reward = -0.1
                accept = False
            else:
                acc = main.evaluate_accuracy(candidate, test_loader, device, max_batches=val_batches)
                params = main.count_parameters(candidate)
                novelty = main.novelty_score(candidate, [main.architecture_signature(model)])
                reward = (acc - last_acc) - lambda_size * np.log(max(1.0, params)) + novelty_weight * novelty
                accept = reward > 0

        policy_opt.zero_grad()
        entropy = -(masked_probs * (masked_probs + 1e-12).log()).sum()
        loss = -(log_probs[action_idx] * reward + entropy_coef * entropy)
        loss.backward()
        policy_opt.step()

        if accept:
            model = candidate
            last_acc = acc
            last_reward = reward
            if last_acc > best_acc + 1e-4:
                best_acc = last_acc
                no_improve = 0
            else:
                no_improve += 1
        else:
            last_reward = reward
            no_improve += 1

        if no_improve >= patience:
            forced = "insert_layer" if rng.rand() < 0.5 else "add_concat"
            candidate, ok = main.apply_mutation_action(model, rng, forced)
            if ok:
                candidate.to(device)
                if main.is_valid_model(candidate, device):
                    model = candidate
                    _ = main.train_brief(model, train_loader, device, steps=training_steps)
                    last_acc = main.evaluate_accuracy(model, test_loader, device, max_batches=val_batches)
                    last_reward = 0.0
                    no_improve = 0

        print(
            f"Iter {step} | Action: {action} | Reward: {reward:.4f} | Acc: {last_acc:.4f} | Params: {main.count_parameters(model)}"
        )
        print(f"Arch: {main.describe_architecture(model)}")

    return model


if __name__ == "__main__":
    run_single_graph_rl()
