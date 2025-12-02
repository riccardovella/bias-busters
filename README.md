# Bias Busters: Exposing Gender Bias in BERT’s Latent Space - BrainHack Lucca 2025

The rapid adoption of large language models like BERT and GPT has brought not only groundbreaking advances in natural language understanding, but also **serious concerns about the social biases** encoded in these models, especially *gender bias*. Today’s AI models are often found **amplifying stereotypes** — associating certain professions or attributes with men and women in ways that reflect, and sometimes exaggerate, societal prejudices. Such biases can seep into everything from automated hiring tools to everyday digital assistants, making the task of understanding and mitigating these flaws an urgent one for both researchers and society at large.

Our project aims to shine a light on these hidden biases—specifically those around gender—in state-of-the-art language models, and to rigorously test strategies for reducing them. We are particularly interested in the subtle, “latent” biases that are baked into the internal representations of words and sentences within models like BERT, and which persist even when no explicit prompting is used.

Using innovative methods inspired by previous research, our team will first **measure gender bias** by examining how likely BERT is to associate different professions with “he” or “she” using carefully crafted test sentences. We then take this a step further by developing a mathematical technique to **"tune" the model’s internal representations**, applying a global transformation aimed at neutralizing these biases.

Then, we will **test how removing occupational gender stereotypes affects other types of bias** — such as those related to family roles or personal traits — to explore whether bias reductions achieved in one context carry over to others. This could reveal whether gender bias has a universal structure in AI or is actually more fragmented than previously thought.
Ultimately, we lay the bases for a *general computational model of stereotypes*, that could possibly be evaluated through human social experiments.

## Meaningful References
- [Bolukbasi, Tolga, et al. "Man is to computer programmer as woman is to homemaker? debiasing word embeddings." ](https://arxiv.org/abs/1607.06520)
- [Kurita, Keita, et al. "Measuring bias in contextualized word representations."](https://arxiv.org/abs/1906.07337)

- [Fersini, Elisabetta, Antonio Candelieri, and Lorenzo Pastore. "On the Generalization of Projection-Based Gender Debiasing in Word Embedding."](https://aclanthology.org/2023.ranlp-1.38.pdf)

## Project Aims
- **Quantify gender bias** in BERT's occupational embeddings using log-probability bias scores (a rigorous metric from [Kurita et al., 2019](https://arxiv.org/abs/1906.07337))
- Learn an optimal linear transformation that remaps BERT's latent space to **reduce gender bias** while preserving semantic information
- **Test generalization** across multiple domains (occupations, family roles, personality traits, sentiment) to ensure the debiasing method is robust

## Project Steps
### Step 1: Replicate the results of Kurita et al. (2019) to compute the bias
1. Prepare a template sentence e.g.“$[TARGET] is a [ATTRIBUTE]$”
2. Replace $[TARGET]$ with $[MASK]$ and compute $p_{tgt}=P([MASK]=[TARGET]| sentence)$
3. Replace both $[TARGET]$ and $[ATTRIBUTE]$ with $[MASK]$, and compute prior probability $p_{prior}=P([MASK]=[TARGET]| sentence)$
4. Compute the association as $log(\frac{p_{tgt}}{p_{prior}})$
   
We refer to this normalized measure of association as the *increased log probability* score and the difference between the *increased log probability* scores for two targets (e.g. he/she) as *log probability bias* score which we use as measure of bias.

In practice, for a word or sentence embedding $e_i \in \mathbb{R}^d$ from BERT, define the log-probability bias score for a template like “TARGET is ATTRIBUTE” as:

$$
b(e_i, \text{attr}) =
\log \frac{P(\text{TARGET}=\text{he} \mid e_i, \text{attr})}
         {P(\text{TARGET}=\text{he} \mid \text{context-only})} -
\log \frac{P(\text{TARGET}=\text{she} \mid e_i, \text{attr})}
         {P(\text{TARGET}=\text{she} \mid \text{context-only})}
$$

This measures asymmetry in how BERT associates “he” vs. “she” with an attribute given the embedding $e_i$.

### Step 2: Learn transformation $W$ to minimize bias

We learn a linear transformation:

$$
e_i' = W e_i, \quad W \in \mathbb{R}^{d \times d}
$$

such that applying $W$ reduces bias. Formalize this as an optimization problem:

$$
\min_{W} \; \mathcal{L}_{\text{bias}}(W) + \lambda \|W - I\|_F^2
$$

where:

- **Bias loss:**

$$
\mathcal{L}_{\text{bias}}(W)
= \sum_{i \in \text{train}} \left| b(W e_i, \text{attr}_i) \right|^2
$$

measures total bias over a set of training pairs (occupation words/sentences).

- **Regularization:** $\lambda \|W - I\|_F^2$ penalizes deviation from identity to preserve semantic content.

**Alternative hybrid approach:** With explicit equalization constraints from word pairs (e.g., "man-nurse" and "woman-nurse" should have equal cosine similarity after transformation), we can solve for an initial $W_0$ in closed form (as outlined in the previous answer), then fine-tune via gradient descent on $L_{bias}$.

### Step 3: Cross-domain evaluation

After training $W$ on occupational roles, we evaluate on:
- Gendered but non-occupational contexts (e.g., family roles, personality traits).
- Non-gendered domains (e.g., sentiment words, neutral descriptors).

Measure:
- Within-domain bias reduction: Does $W$ reduce log-probability bias on held-out occupations?
- Cross-domain transfer: Does it reduce bias in other gendered contexts?
- Semantic preservation: Do downstream tasks (e.g., sentence similarity, classification) maintain performance?
