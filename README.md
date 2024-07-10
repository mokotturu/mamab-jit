# mamab-jit

This repo contains all the code for my master's thesis, "Enhancing Team Performance in Multi-Agent Multi-Armed Bandits through Optimization". All experiments are implemented in Python using NumPy, SciPy, Matplotlib, NetworkX, CVXPY, and Numba. Numba is a just-in-time (JIT) compiler that optimizes the performance of Python and NumPy code by translating it to machine code. The experiments in this repo take advantage of the "no-python" mode in Numba along with parallel processing to speed up experiments. See [here](https://numba.pydata.org/) for more details on Numba.

<ins>**Note: This repo contains a lot of old code that was used for testing early ideas, quick experimentation, and graphing. They may contain unfinished code, repetitive code, bugs, or other issues. They are moved to a separate directory `old_code` for reference.**</ins>

<hr />

The main code for the experiments in the thesis is in the following files:

- `bandit_rating_proto.ipynb`
  - Initial experiments to measure and analyze the difficulty of bandits (contains some plots used in the thesis)
- `bandit_rating_plots.ipynb`
  - Calculate and generate plots for the bandit difficulty rating measure
- `competency_graphs.ipynb`
  - Analyze performances when agents in a team have different competencies
- `coopucb2_competency.py`
  - Run CoopUCB2 with teams containing agents with different competencies while playing bandits from various difficulty levels
  - Run using `python3 coopucb2_competency.py --network NETWORK_NAME --alg ALG`
    - Refer to `data/saved_networks` for available graph structures/networks
    - `ALG` can either be `coopucb2_og` or `coopucb2_limited_communication`
- `graph_optimization.py`
  - Contains implementations of the heuristic and optimization methods for edge weight setting of a graph.
- `large_nets.ipynb` and `nets.ipynb`
  - Generate and analyze large networks
- `long_optimization.ipynb`
  - Run the long-term optimization process proposed by the thesis

# Setup project

1. Setup a virtual environment in the root directory of the repo
     - `python3 -m venv venv`
2. Activate the virtual environment
     - `source venv/bin/activate`
3. Install the required packages
     - `pip3 install -r requirements.txt`
4. Run the experiments
