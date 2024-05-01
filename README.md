# RISC
In the real world, the strong episode resetting mechanisms that are needed to train
agents in simulation are unavailable. The resetting assumption limits the potential of
reinforcement learning in the real world, as providing resets to an agent usually
requires the creation of additional handcrafted mechanisms or human interventions.
Recent work aims to train agents (forward) with learned resets by constructing a second
(backward) agent that returns the forward agent to the initial state. We find that the
termination and timing of the transitions between these two agents are crucial for
algorithm success. With this in mind, we create a new algorithm, Reset Free RL with
Intelligently Switching Controller (RISC) which intelligently switches between the two
agents based on the agent’s confidence in achieving its current goal. Our new method
achieves state-of-the-art performance on several challenging environments for reset-free
RL.

## Installation
To install all the necessary dependencies, inside of a virtual environment (Python 3.10), first install the EARL benchmark:

```
git clone https://github.com/dapatil211/earl_benchmark.git
cd earl_benchmark
pip install -e .
```

Next, install the rest of the dependencies in `requirements.txt`.
```
cd ..
pip install -r requirements.txt
```

## Running the code
The configs for running the RISC agents described in the paper can be found in the
folder `risc/configs/`. To launch an agent, go to the `risc/` directory and run:
```
cd risc/
python main.py -c configs/risc_sd.yaml
```

We provide RISC configs for Sawyer Door, Sawyer Peg, Tabletop Manipulation, Minitaur,
and Minigrid.

## Creating the Visualizations
The current codebase logs to Wandb. We provide the raw numbers for our runs as well
as the code to convert them into the plots seen in the paper. To create all the plots,
simply run:
```
cd visualization
python create_figures.py
```

To create just one of the figures, run:
```
cd visualization
python create_figures.py -c <"figure_3_top" | "figure_3_bottom" | "figure_4_left" | "figure_4_right" | "figure_5" | "figure_6">
```

## Bibtex
If you found our work useful, please cite our paper.

```
@inproceedings{
    patil2024intelligent,
    title={Intelligent Switching for Reset-Free {RL}},
    author={Darshan Patil and Janarthanan Rajendran and Glen Berseth and Sarath Chandar},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=Nq45xeghcL}
}
```
