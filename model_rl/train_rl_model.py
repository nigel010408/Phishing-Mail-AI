import numpy as np
import random
import pickle
import os
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# load from huggingface if possible
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# load vector
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MAX_FEATURES = 500


def load_or_create_vectorizer(texts=None):
    """Load a pre-saved TF-IDF vectorizer, or create + fit a new one."""
    if os.path.exists(VECTORIZER_PATH):
        print(f"[Vectorizer] Loading from '{VECTORIZER_PATH}' …")
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        print("[Vectorizer] Fitting new TF-IDF vectorizer …")
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        if texts is not None:
            vectorizer.fit(texts)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"[Vectorizer] Saved to '{VECTORIZER_PATH}'.")
    return vectorizer


# example dataset if hugginface fails
FALLBACK_EMAILS = [
    ("Congratulations! You've won a $1,000 gift card. Click here to claim now!", 1),
    ("Your bank account has been suspended. Verify your identity immediately.", 1),
    ("URGENT: Confirm your PayPal details or your account will be closed.", 1),
    ("You have a pending wire transfer. Login to approve or reject the funds.", 1),
    ("Dear user, your password expires today. Reset it now to avoid lockout.", 1),
    ("FREE iPhone giveaway – limited time! Enter your credit card to pay shipping.", 1),
    ("IRS Notice: You owe back taxes. Pay within 24 hours to avoid arrest.", 1),
    ("Your Netflix subscription has been cancelled. Update billing info here.", 1),
    ("Security alert: Unusual sign-in on your Microsoft account. Verify now.", 1),
    ("You've been selected for a prize. Send your SSN to claim your reward.", 1),
    ("Delivery failed. Pay $1.99 re-delivery fee via secure link below.", 1),
    ("We detected suspicious activity. Provide your login to restore access.", 1),
    ("Hi team, please find attached the Q3 financial report for your review.", 0),
    ("Reminder: The weekly all-hands meeting is scheduled for Friday at 10 AM.", 0),
    ("Your Amazon order #112-3456789 has shipped and will arrive Thursday.", 0),
    ("Thank you for signing up! Here's how to get started with our service.", 0),
    ("Your monthly bank statement is now available in online banking.", 0),
    ("Meeting notes from yesterday's product sync have been shared on Drive.", 0),
    ("Hi, just following up on the proposal I sent last week. Let me know.", 0),
    ("Your flight booking confirmation: NYC → LAX on March 15 at 8:00 AM.", 0),
    ("Newsletter: Top 10 Python tips from the dev community this month.", 0),
    ("Invoice #4892 from Acme Corp is due on April 1st. Thank you.", 0),
    ("Invitation: John's birthday party this Saturday at 7 PM. RSVP below.", 0),
    ("Your GitHub pull request #47 was approved and merged into main.", 0),
]


def load_dataset_hf():
    """Try to load the HuggingFace phishing dataset; fall back to examples."""
    if not HF_AVAILABLE:
        print("[Dataset] 'datasets' package not installed. Using fallback data.")
        return None, None

    try:
        print("[Dataset] Downloading drorrabin/phishing_emails-data from HuggingFace …")
        ds = load_dataset("drorrabin/phishing_emails-data", split="train")
        
        text_col = "text" if "text" in ds.column_names else ds.column_names[0]
        label_col = "label" if "label" in ds.column_names else ds.column_names[-1]
        texts = ds[text_col]
        labels = ds[label_col]
        
        unique = list(set(labels))
        if len(unique) == 2 and not all(isinstance(l, int) for l in labels):
            mapping = {unique[0]: 0, unique[1]: 1}
            labels = [mapping[l] for l in labels]
        print(f"[Dataset] Loaded {len(texts)} samples.")
        return list(texts), list(labels)
    except Exception as e:
        print(f"[Dataset] HuggingFace load failed ({e}). Using fallback data.")
        return None, None


def get_dataset():
    texts_hf, labels_hf = load_dataset_hf()
    if texts_hf is not None:
        return texts_hf, labels_hf
    
    texts, labels = zip(*FALLBACK_EMAILS)
    return list(texts), list(labels)


# split data

def prepare_data(vectorizer, texts, labels, test_size=0.2, random_state=42):
    """Vectorize text and split into train/test arrays."""
    X = vectorizer.transform(texts).toarray().astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


class PhishingEnv:
    """
    Custom RL environment for phishing email classification.

    - State  : TF-IDF feature vector of the current email.
    - Action : 0 → predict legitimate | 1 → predict phishing.
    - Reward : +1 for correct classification, -1 for incorrect.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.state_size = X.shape[1]
        self.action_size = 2
        self.current_idx = 0
        self.indices = np.arange(self.n_samples)
        np.random.shuffle(self.indices)

    def reset(self):
        """Shuffle and restart from the first sample."""
        np.random.shuffle(self.indices)
        self.current_idx = 0
        return self.X[self.indices[self.current_idx]]

    def step(self, action: int):
        """
        Apply action, compute reward, advance to next sample.

        Returns: (next_state, reward, done)
        """
        true_label = self.y[self.indices[self.current_idx]]
        reward = 1.0 if action == true_label else -1.0

        self.current_idx += 1
        done = self.current_idx >= self.n_samples

        if done:
            next_state = self.reset()
        else:
            next_state = self.X[self.indices[self.current_idx]]

        return next_state, reward, done

    def sample_action(self) -> int:
        return random.randint(0, self.action_size - 1)


# DQN model

class DQNModel(nn.Module):
    """
    Deep Q-Network: maps state → Q-values for each action.

    Architecture: FC → BN → ReLU → Dropout → FC → BN → ReLU → FC
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes=(256, 128)):
        super(DQNModel, self).__init__()

        layers = []
        in_size = state_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            ]
            in_size = h
        layers.append(nn.Linear(in_size, action_size))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# DQN agent

class DQNAgent:
    """
    DQN agent with experience replay and a target network.

    Hyperparameters
    ---------------
    gamma       : discount factor (future rewards)
    epsilon     : initial exploration rate
    epsilon_min : minimum exploration rate
    epsilon_decay: per-step multiplicative decay
    lr          : Adam learning rate
    batch_size  : mini-batch size for replay
    memory_size : max replay buffer capacity
    target_update_freq : steps between target network syncs
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lr: float = 1e-3,
        batch_size: int = 64,
        memory_size: int = 10_000,
        target_update_freq: int = 200,
        device: str = "cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.step_count = 0

        self.memory: deque = deque(maxlen=memory_size)

        self.policy_net = DQNModel(state_size, action_size).to(self.device)
        self.target_net = DQNModel(state_size, action_size).to(self.device)
        self._sync_target()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        self.policy_net.train()
        return int(q_values.argmax(dim=1).item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._sync_target()

        return loss.item()

    def _sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str = "dqn_phishing.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"[Agent] Model saved → '{path}'")

    def load(self, path: str = "dqn_phishing.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self._sync_target()
        print(f"[Agent] Model loaded ← '{path}'")


# model training

def train(
    agent: DQNAgent,
    env: PhishingEnv,
    n_episodes: int = 50,
    print_every: int = 10,
) -> list:
    """
    Train the DQN agent on the phishing environment.

    One *episode* = one full pass through the shuffled training set.

    Returns a list of (episode, avg_reward, epsilon, avg_loss) tuples.
    """
    history = []
    print("\n" + "=" * 60)
    print("  TRAINING  –  DQN Phishing Detector")
    print("=" * 60)

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.replay()
            if loss is not None:
                total_loss += loss
                loss_count += 1

        avg_reward = total_reward / env.n_samples
        avg_loss   = total_loss / loss_count if loss_count else 0.0
        history.append((episode, avg_reward, agent.epsilon, avg_loss))

        if episode % print_every == 0 or episode == 1:
            print(
                f"  Ep {episode:>4}/{n_episodes} │ "
                f"AvgReward: {avg_reward:+.4f} │ "
                f"ε: {agent.epsilon:.4f} │ "
                f"Loss: {avg_loss:.6f}"
            )

    print("=" * 60 + "\n")
    return history


# evaluation

def print_results_table(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Print a classification report table styled after the reference image:

        ┌──────────────┬───────────┬────────┬──────────┬─────────┐
        │              │ precision │ recall │ F1-score │ Support │
        ├──────────────┼───────────┼────────┼──────────┼─────────┤
        │ 0            │      0.99 │   1.00 │     1.00 │    3369 │
        │ 1            │      0.97 │   0.95 │     0.96 │     336 │
        │ accuracy     │           │        │     0.99 │    3705 │
        │ macro avg    │      0.98 │   0.97 │     0.98 │    3705 │
        │ weighted avg │      0.99 │   0.99 │     0.99 │    3705 │
        └──────────────┴───────────┴────────┴──────────┴─────────┘
    """
    from sklearn.metrics import precision_recall_fscore_support

    labels_uniq = sorted(np.unique(y_test))
    n_total     = len(y_test)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_uniq
    )

    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    acc_val = accuracy_score(y_test, y_pred)

    col0_w  = 14
    num_w   = 11
    sup_w   = 9

    hline_top = (
        "┌" + "─" * col0_w + "┬" + "─" * num_w + "┬" + "─" * num_w +
        "┬" + "─" * num_w + "┬" + "─" * sup_w + "┐"
    )
    hline_mid = (
        "├" + "─" * col0_w + "┼" + "─" * num_w + "┼" + "─" * num_w +
        "┼" + "─" * num_w + "┼" + "─" * sup_w + "┤"
    )
    hline_bot = (
        "└" + "─" * col0_w + "┴" + "─" * num_w + "┴" + "─" * num_w +
        "┴" + "─" * num_w + "┴" + "─" * sup_w + "┘"
    )

    def row(label, p, r, f, s, blank_pr=False):
        p_str = f"{p:.2f}".rjust(num_w - 2) if not blank_pr else " " * (num_w - 2)
        r_str = f"{r:.2f}".rjust(num_w - 2) if not blank_pr else " " * (num_w - 2)
        f_str = f"{f:.2f}".rjust(num_w - 2)
        s_str = str(int(s)).rjust(sup_w - 2)
        label_str = label.ljust(col0_w - 2)
        return (
            f"│ {label_str} │ {p_str} │ {r_str} │ {f_str} │ {s_str} │"
        )

    header = (
        f"│ {'':>{col0_w - 2}} │ {'precision':>{num_w - 2}} │"
        f" {'recall':>{num_w - 2}} │ {'F1-score':>{num_w - 2}} │"
        f" {'Support':>{sup_w - 2}} │"
    )

    print()
    print(hline_top)
    print(header)
    print(hline_mid)
    for i, lbl in enumerate(labels_uniq):
        print(row(str(lbl), prec[i], rec[i], f1[i], sup[i]))
    print(hline_mid)
    print(row("accuracy",     acc_val, acc_val, acc_val, n_total, blank_pr=True))
    print(row("macro avg",    prec_mac, rec_mac, f1_mac, n_total))
    print(row("weighted avg", prec_w,   rec_w,   f1_w,   n_total))
    print(hline_bot)
    print()


def evaluate(agent: DQNAgent, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate the trained agent greedily (ε=0) on held-out test data.

    Prints accuracy, confusion matrix, and a styled classification table.
    """
    print("=" * 60)
    print("  EVALUATION  –  Test Set")
    print("=" * 60)

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    predictions = []
    agent.policy_net.eval()
    with torch.no_grad():
        for state in X_test:
            action = agent.act(state)
            predictions.append(action)
    agent.policy_net.train()

    agent.epsilon = saved_epsilon

    y_pred = np.array(predictions)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")

    print_results_table(y_test, y_pred)

    print("=" * 60 + "\n")

    report = classification_report(y_test, y_pred)
    return acc, cm, report


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Config] Device: {DEVICE.upper()}")

    texts, labels = get_dataset()

    vectorizer = load_or_create_vectorizer(texts)

    X_train, X_test, y_train, y_test = prepare_data(vectorizer, texts, labels)

    STATE_SIZE  = X_train.shape[1]
    ACTION_SIZE = 2               

    train_env = PhishingEnv(X_train, y_train)

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.993,
        lr=1e-3,
        batch_size=64,
        memory_size=10_000,
        target_update_freq=300,
        device=DEVICE,
    )

    history = train(agent, train_env, n_episodes=60, print_every=10)

    agent.save("dqn_phishing.pth")

    accuracy, cm, report = evaluate(agent, X_test, y_test)

    print("[Summary] Training reward progression (every 10 episodes):")
    for ep, avg_r, eps, loss in history[::10]:
        bar = "█" * int((avg_r + 1) * 20)
        print(f"  Ep {ep:>3} │ {bar:<40} │ {avg_r:+.3f}")
    print()