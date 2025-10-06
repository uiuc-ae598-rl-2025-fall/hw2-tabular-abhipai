import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import os

# --- Helper Functions for Plotting ---

def plot_learning_curves(results, title, save_path=None):
    """Plots the evaluation return vs. time steps for multiple algorithms and optionally saves to disk."""
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['timesteps'], result['eval_returns'], label=name)
    plt.title(title, fontsize=16)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Average Evaluation Return', fontsize=12)
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_policy_and_value_function(Q, title, save_path=None):
    """Visualizes the learned policy and state-value function and optionally saves to disk."""
    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)

    V_grid = V.reshape(4, 4)
    policy_grid = policy.reshape(4, 4)

    action_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_arrows = np.array([[action_map[action] for action in row] for row in policy_grid])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(V_grid, annot=policy_arrows, fmt='', cmap='viridis', cbar=True,
                linewidths=.5, linecolor='gray', ax=ax, annot_kws={"size": 20})
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# --- Evaluation Function ---

def evaluate_policy(env, Q, num_episodes=300, gamma=0.95, base_seed=12345):
    """Evaluates a greedy policy derived from Q-values using a fixed seed schedule."""
    total_returns = 0
    for i in range(num_episodes):
        terminated, truncated = False, False
        state, _ = env.reset(seed=base_seed + i)
        episode_return = 0
        t = 0
        while not (terminated or truncated):
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += (gamma ** t) * reward
            t += 1
        total_returns += episode_return
    return total_returns / num_episodes

# --- Learning Algorithms from Sutton and Barto ---

def on_policy_first_visit_mc_control(env, num_episodes, gamma=0.95, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                                     eval_env=None, eval_interval=1000, eval_episodes=300, base_seed=0):
    """
    On-policy first-visit MC control algorithm.
    Based on Sutton and Barto, Reinforcement Learning, Chapter 5.4[cite: 1945].
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions)) # For incremental averaging
    epsilon = epsilon_start

    eval_returns_history = []
    timesteps_history = []
    total_timesteps = 0

    for episode_num in tqdm(range(num_episodes), desc="MC Training"):
        # Generate an episode
        episode = []
        state, _ = env.reset(seed=base_seed + episode_num)
        terminated, truncated = False, False
        while not (terminated or truncated):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_timesteps += 1

        # Learn from the episode
        G = 0
        visited_sa_pairs = set()
        # Loop backward through the episode [cite: 1943]
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            sa_pair = (state, action)
            # First-visit MC check
            if sa_pair not in visited_sa_pairs:
                visited_sa_pairs.add(sa_pair)
                N[state, action] += 1
                # Incremental update of Q-value [cite: 1492, 1943]
                Q[state, action] += (1 / N[state, action]) * (G - Q[state, action])

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Periodic evaluation
        if (eval_env is not None) and (episode_num % eval_interval == 0):
            eval_return = evaluate_policy(eval_env, Q, num_episodes=eval_episodes, gamma=gamma, base_seed=base_seed + 10_000_000)
            eval_returns_history.append(eval_return)
            timesteps_history.append(total_timesteps)

    return Q, {'timesteps': timesteps_history, 'eval_returns': eval_returns_history}

def sarsa(env, num_episodes, alpha=0.5, gamma=0.95, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01,
          eval_env=None, eval_interval=1000, eval_episodes=300, base_seed=0):
    """
    SARSA (on-policy TD(0)) control algorithm.
    Based on Sutton and Barto, Reinforcement Learning, Chapter 6.4[cite: 1975].
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    epsilon = epsilon_start

    eval_returns_history = []
    timesteps_history = []
    total_timesteps = 0

    for episode_num in tqdm(range(num_episodes), desc="SARSA Training"):
        state, _ = env.reset(seed=base_seed + episode_num)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        terminated, truncated = False, False
        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_timesteps += 1

            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            # SARSA update rule [cite: 1974]
            td_target = reward + gamma * Q[next_state, next_action] * (not terminated)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state, action = next_state, next_action

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Periodic evaluation
        if (eval_env is not None) and (episode_num % eval_interval == 0):
            eval_return = evaluate_policy(eval_env, Q, num_episodes=eval_episodes, gamma=gamma, base_seed=base_seed + 10_000_000)
            eval_returns_history.append(eval_return)
            timesteps_history.append(total_timesteps)

    return Q, {'timesteps': timesteps_history, 'eval_returns': eval_returns_history}


def q_learning(env, num_episodes, alpha=0.5, gamma=0.95, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01,
               eval_env=None, eval_interval=1000, eval_episodes=300, base_seed=0):
    """
    Q-learning (off-policy TD(0)) control algorithm.
    Based on Sutton and Barto, Reinforcement Learning, Chapter 6.5[cite: 1976].
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    epsilon = epsilon_start

    eval_returns_history = []
    timesteps_history = []
    total_timesteps = 0

    for episode_num in tqdm(range(num_episodes), desc="Q-Learning Training"):
        state, _ = env.reset(seed=base_seed + episode_num)
        terminated, truncated = False, False

        while not (terminated or truncated):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_timesteps += 1
            
            # Q-learning update rule 
            best_next_action_value = np.max(Q[next_state])
            td_target = reward + gamma * best_next_action_value * (not terminated)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Periodic evaluation
        if (eval_env is not None) and (episode_num % eval_interval == 0):
            eval_return = evaluate_policy(eval_env, Q, num_episodes=eval_episodes, gamma=gamma, base_seed=base_seed + 10_000_000)
            eval_returns_history.append(eval_return)
            timesteps_history.append(total_timesteps)

    return Q, {'timesteps': timesteps_history, 'eval_returns': eval_returns_history}


# --- Main Execution Block ---

def run_experiment(is_slippery):
    """Runs all algorithms for a given environment setting and plots results."""
    
    map_desc = ["SFFF", "FHFH", "FFFF", "HFFG"]
    env_id = 'FrozenLake-v1'
    
    # Global seed for reproducibility
    seed = 0
    np.random.seed(seed)

    # Training schedules tuned for convergence
    if is_slippery:
        episodes = 200000
        eval_interval = 2000
        eval_episodes = 300
        epsilon_decay = 0.999
        epsilon_min = 0.05
        alpha = 0.2
    else:
        episodes = 20000
        eval_interval = 1000
        eval_episodes = 300
        epsilon_decay = 0.995
        epsilon_min = 0.01
        alpha = 0.5

    env = gym.make(env_id, desc=map_desc, is_slippery=is_slippery)
    eval_env = gym.make(env_id, desc=map_desc, is_slippery=is_slippery)

    # Run algorithms
    q_mc, history_mc = on_policy_first_visit_mc_control(
        env,
        episodes,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        eval_env=eval_env,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        base_seed=seed,
    )

    q_sarsa, history_sarsa = sarsa(
        env,
        episodes,
        alpha=alpha,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        eval_env=eval_env,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        base_seed=seed,
    )

    q_q_learning, history_q_learning = q_learning(
        env,
        episodes,
        alpha=alpha,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        eval_env=eval_env,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        base_seed=seed,
    )
    
    env.close()
    eval_env.close()

    # Plot learning curves
    results = {
        'On-Policy MC': history_mc,
        'SARSA': history_sarsa,
        'Q-Learning': history_q_learning,
    }
    plot_title = f'Learning Curves on Frozen Lake {"(Slippery)" if is_slippery else "(Non-Slippery)"}'
    curves_path = os.path.join('outputs', f'curves_{"slippery" if is_slippery else "non_slippery"}.png')
    plot_learning_curves(results, plot_title, save_path=curves_path)

    # Plot policies and value functions
    plot_policy_and_value_function(q_mc, f'MC Policy and V(s) {"(Slippery)" if is_slippery else "(Non-Slippery)"}',
                                   save_path=os.path.join('outputs', f'policy_value_mc_{"slippery" if is_slippery else "non_slippery"}.png'))
    plot_policy_and_value_function(q_sarsa, f'SARSA Policy and V(s) {"(Slippery)" if is_slippery else "(Non-Slippery)"}',
                                   save_path=os.path.join('outputs', f'policy_value_sarsa_{"slippery" if is_slippery else "non_slippery"}.png'))
    plot_policy_and_value_function(q_q_learning, f'Q-Learning Policy and V(s) {"(Slippery)" if is_slippery else "(Non-Slippery)"}',
                                   save_path=os.path.join('outputs', f'policy_value_q_learning_{"slippery" if is_slippery else "non_slippery"}.png'))


if __name__ == '__main__':
    print("--- Running Experiment for Non-Slippery Frozen Lake ---")
    run_experiment(is_slippery=False)
    
    print("\n--- Running Experiment for Slippery Frozen Lake ---")
    run_experiment(is_slippery=True)