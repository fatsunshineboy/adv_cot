# \[adv_CoT]: Code Implementation

> Corresponding Paper: Chain-of-Thought Prompt Optimization via Adversarial Learning
>
> Paper Link: [Chain-of-Thought Prompt Optimization via Adversarial Learning[v1\] | Preprints.org](https://www.preprints.org/manuscript/202510.2100)

## 1. Project Overview

* adv-CoT is a prompt optimization framework that integrates adversarial learning into Chain-of-Thought (CoT) prompting to enhance LLM reasoning without modifying model parameters.
* It leverages generator–discriminator interactions, feedback revision, and verification to improve reasoning robustness and accuracy.



## 2. Environment Setup

### 2.1 Dependency List

It is recommended to manage dependencies with `requirements.txt`. List core dependencies here:

```
openai==2.8.1
matplotlib==3.10.7
numpy==2.3.5
scipy==1.16.3
```

### 2.2 Installation Steps

```
\# 1. Clone the repository

git clone git@github.com:fatsunshineboy/adv_cot.git

cd adv_cot

\# 2. Create a virtual environment (optional but recommended)

conda create -n adv_cot python=3.12

conda activate adv_cot

\# 3. Install dependencies

pip install -r requirements.txt
```



## 3. Quick Start

```python
python -m adv_cot.main
# Experimental logs and results are saved in the log_adv_cot folder under the running directory.
```



## 4. Contact Information

* Author Email: \[yangguang@whu.edu.cn]

* Issue Reporting: Welcome to submit bugs or suggestions via GitHub Issues