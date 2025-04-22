## Introduction

Reinforcement Learning (RL) has become a cornerstone in optimizing large language models (LLMs) for reasoning-intensive tasks, such as mathematical problem-solving and code generation. Among the RL algorithms used in LLM training, GRPO (Group Relative Policy Optimization), DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization), and VAPO (Value Augmented Proximal Policy Optimization) represent three distinct approaches to policy optimization. This report provides a detailed explanation of how these algorithms work, their mechanisms, and their applications in LLM training.

## GRPO: Group Relative Policy Optimization

### Overview
GRPO is a value-free RL algorithm that eliminates the need for a value model and instead computes advantages based on group reward normalization. It is particularly suited for scenarios where training a reliable value model is challenging due to instability or computational overhead ([arXiv](https://arxiv.org/pdf/2504.05118)).

### Key Mechanisms
1. **Group-Based Advantage Calculation**:
   - GRPO samples a group of responses for each prompt and calculates the advantage by normalizing the rewards within the group.
   - The advantage for each response is computed as:
     ```
     A_i = (R_i - mean(R_group)) / std(R_group)
     ```
     where `R_group` represents the rewards of all responses in the group ([arXiv](https://arxiv.org/pdf/2504.05118)).

2. **Clipped Objective**:
   - GRPO employs a clipped surrogate objective similar to PPO (Proximal Policy Optimization) to stabilize training:
     ```
     J_GRPO = E[min(r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t)]
     ```
     where `r_t` is the importance sampling ratio, and `ε` is the clipping range ([arXiv](https://arxiv.org/pdf/2504.05118)).

3. **KL Penalty Term**:
   - GRPO includes a KL divergence penalty to regulate the divergence between the online policy and the reference policy, ensuring alignment with the initial model ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).

### Advantages and Limitations
- **Advantages**:
  - Simplifies computation by avoiding value model training.
  - Provides a stable baseline for advantage calculation in complex tasks.
- **Limitations**:
  - Lacks the precision of value-model-based methods in credit assignment.
  - May struggle with long-chain-of-thought (CoT) reasoning tasks due to its reliance on group-level reward aggregation.

## DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization

### Overview
DAPO builds upon GRPO by introducing several enhancements to address entropy collapse, reward noise, and training instability. It is designed for large-scale RL systems and achieves state-of-the-art performance in reasoning tasks ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).

### Key Mechanisms
1. **Clip-Higher**:
   - Decouples the lower and upper clipping ranges (`ε_low` and `ε_high`) to promote exploration and avoid entropy collapse.
   - Higher clipping values allow low-probability tokens to increase their probabilities, enhancing diversity ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).

2. **Dynamic Sampling**:
   - Filters out prompts with zero or full accuracy to maintain effective gradients for policy updates.
   - Ensures a consistent number of prompts in each batch, reducing gradient variance and improving training stability ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).

3. **Token-Level Policy Gradient Loss**:
   - Replaces sample-level loss with token-level loss to balance the contribution of long and short sequences.
   - Prevents long sequences from disproportionately influencing the overall loss ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).

4. **Overlong Reward Shaping**:
   - Introduces penalties for excessively long responses to stabilize training and improve performance.
   - Uses a soft punishment mechanism to signal the model to avoid overly lengthy outputs ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).

### Advantages and Limitations
- **Advantages**:
  - Addresses entropy collapse and reward noise effectively.
  - Improves training efficiency and stability.
  - Enhances performance in long-CoT reasoning tasks.
- **Limitations**:
  - Requires careful tuning of hyperparameters (e.g., clipping ranges, reward shaping thresholds).
  - Computationally more intensive than GRPO.

## VAPO: Value Augmented Proximal Policy Optimization

### Overview
VAPO is a value-model-based RL framework that outperforms value-free methods like GRPO and DAPO in long-CoT reasoning tasks. It combines techniques from previous works, such as VC-PPO and DAPO, to address challenges in value model bias, heterogeneous sequence lengths, and reward sparsity ([arXiv](https://arxiv.org/pdf/2504.05118)).

### Key Mechanisms
1. **Length-Adaptive GAE**:
   - Dynamically adjusts the λ parameter in Generalized Advantage Estimation (GAE) based on response lengths.
   - Optimizes the bias-variance trade-off for sequences of varying lengths ([arXiv](https://arxiv.org/pdf/2504.05118)).

2. **Decoupled GAE Computation**:
   - Separates advantage computation for the value and policy networks.
   - Uses λ = 1.0 for value updates to reduce bias and λ < 1.0 for policy updates to accelerate convergence ([arXiv](https://arxiv.org/pdf/2504.05118)).

3. **Token-Level Loss**:
   - Similar to DAPO, VAPO employs token-level loss to balance the contribution of long and short sequences ([arXiv](https://arxiv.org/pdf/2504.05118)).

4. **Group Sampling**:
   - Samples multiple responses for each prompt to enhance contrastive signals and improve learning efficiency ([arXiv](https://arxiv.org/pdf/2504.05118)).

### Advantages and Limitations
- **Advantages**:
  - Achieves superior performance in long-CoT reasoning tasks.
  - Provides more precise credit assignment and lower-variance value estimates.
  - Demonstrates high training stability and efficiency.
- **Limitations**:
  - Requires overcoming challenges in value model training, such as initialization bias and reward sparsity.
  - Computationally demanding compared to value-free methods.

## Comparative Analysis

| Algorithm | Approach | Key Features | Strengths | Weaknesses |
|-----------|----------|--------------|-----------|------------|
| GRPO      | Value-Free | Group-Based Advantage, KL Penalty | Simplifies computation, stable baseline | Limited precision, struggles with long-CoT tasks |
| DAPO      | Value-Free | Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Reward Shaping | Addresses entropy collapse, improves stability | Requires careful tuning, computationally intensive |
| VAPO      | Value-Based | Length-Adaptive GAE, Decoupled GAE, Token-Level Loss, Group Sampling | Superior performance, precise credit assignment | Challenging value model training, high computational cost |

### Performance Metrics
- GRPO achieves 47 points on the AIME24 benchmark ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).
- DAPO improves upon GRPO, reaching 50 points with 50% fewer training steps ([DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)).
- VAPO sets a new state-of-the-art score of 60.4 points, demonstrating its efficiency and reliability ([arXiv](https://arxiv.org/pdf/2504.05118)).

## Conclusion

GRPO, DAPO, and VAPO represent three distinct approaches to RL in LLM training, each with unique mechanisms and applications. GRPO provides a simple and stable baseline for advantage calculation, while DAPO introduces innovative techniques to address entropy collapse and reward noise. VAPO, as a value-model-based framework, achieves superior performance in reasoning-intensive tasks by addressing challenges in value model training and sequence length management. Together, these algorithms highlight the evolving landscape of RL methods in optimizing LLMs for complex reasoning tasks.

## References

- VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks. [arXiv](https://arxiv.org/pdf/2504.05118)
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale. [DAPO Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)
