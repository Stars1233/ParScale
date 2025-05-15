<div align="center">


# Parallel Scaling Law for Language Model


_Another Scaling Law beyond Parameters and Inference Time Scaling_

[![Paper](https://img.shields.io/badge/arXiv-2505.xxxxx-red)](https://arxiv.org/abs/2505.xxxxx)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/ParScale)

<p align="center">
    💡&nbsp;<a href="#-key-findings">Key Findings</a>
    | 📈&nbsp;<a href="#-scaling-law">Scaling Law</a>
    | ⚡&nbsp;<a href="#-cost-analysis">Cost Analysis</a>
    | 🔥&nbsp;<a href="#-scaling-data">Scaling Data</a>
    | 📚&nbsp;<a href="#-citation">Citation</a>
</p>
</div>

## 🌟 About

- Most believe that scaling language models requires a heavy cost in either **parameters** (parameter scaling) or **inference time** (inference-time scaling). 
- We introduce the *third* scaling paradigm for scaling LLMs: leverages **parallel computation** during both training and inference time (Parallel Scaling, or *ParScale*).
- We apply $P$ diverse and learnable transformations to the input, execute forward passes of the model in parallel, and dynamically aggregate the $P$ outputs. 
<div align="center">
<img src="figures/teaser.png" style="width: 80%;" />
</div>
---

## 💡 Key Findings
<div align="center">
<img src="figures/scaling_comparison.png" style="width: 80%;" />
</div>

Here are the core insights and benefits distilled from our theoretical analysis and empirical evaluations:

📈 **Logarithmic Scaling Law**: We theoretically and empirically establish that **scaling with $P$ parallel streams is comparable to scaling the number of parameters by** $O(\log P)$. This suggests that parallel computation can serve as an efficient substitute for parameter growth, especially for larger models.

✅ **Universal Applicability**: Unlike inference-time scaling which requires specialized data and limited application, it works with any model architecture, optimization method, data, or downstream task.


🧠 **Stronger Performance on Reasoning Tasks**: Reasoning-intensive tasks (e.g., coding or math) benefit more from ParScale, which suggests that scaling computation can effectively push the boundary of reasoning. 

⚡ **Superior Inference Efficiency**: ParScale can use up to **22x less memory increase and 6x less latency increase** compared to parameter scaling that achieves the same performance improvement (batch size=1).

🧱 **Cost-Efficient Training via Two-Stage Strategy**: Training a parallel-scaled model doesn't require starting from scratch. With a two-stage training strategy, we can post-train ithe parallel components using only a small amount of data.

🔁 **Dynamic Adaptation at Inference Time**: We find that ParScale remains effective with frozen main parameters for different $P$. This illustrates the potential of dynamic parallel scaling: switching $P$ to dynamically adapt model capabilities during inference.

We release the inference code in `modeling_qwen2_parscale.py` and `configuration_qwen2_parscale.py`. Our 67 checkpoints is available at [🤗 HuggingFace](https://huggingface.co/ParScale) (coming soon).

---

## 📈 Scaling Law

- Our preliminary theoretical analysis suggests that the loss of ParScale may follow a power law similar to the Chinchilla scaling law. 
- We then carry out large-scale pre-training experiments on the Stack-V2 and Pile corpus, by ranging $P$ from 1 to 8 and model parameters from 500M to 4.4B. 
- We use the results to fit a new *parallel scaling law* that generalizes the Chinchilla scaling law.
- We release our parametric fitting code in `parametric_fit.py`.
<div align="center">
<img src="figures/scaling_law.png" style="width: 70%;" />

<img src="figures/scaling_law2.png" style="width: 70%;" />
</div>
---

## ⚡ Cost Analysis

<div align="center">
<img src="figures/cost.png" style="width: 70%;" />
</div>

- We further compare the inference efficiency between parallel scaling and parameter scaling at equivalent performance levels. 
- We release our analysis code in `cost_analysis.py`. Before using it, you should first install [llm-analysis](https://github.com/cli99/llm-analysis):

```bash
git clone https://github.com/cli99/llm-analysis.git
cd llm-analysis
pip install .
```

- You can use the following command to analyze the inference memory and latency cost for our 4.4B model, with $P=2$ and batch size=2:
```bash
python cost_analysis.py --hidden_size 2560 --intermediate_size 13824 --P 2 --batch_size 2
```

---

## 🔥 Scaling Data

!under construction!

This section and model checkpoints are currently under development. Stay tuned for updates!



## 📚 Citation

!under construction!
