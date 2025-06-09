# SIERL

## Installation

To install all the necessary dependencies, create a Conda virtual environment using the file provided:

```
git clone https://github.com/gsotirchos/RISC
conda env create -f environment.yaml
```


## Running the code

The configs for running the RISC agents described in the paper can be found in the
folder `risc/configs/`. To launch an agent, go to the `risc/` directory and run:

```
cd risc/
python main.py -c configs/episodic/sierl.yaml
```

We provide SIERL configs for Minigrid and the Atari game Montezuma's Revenge.


<!--
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
-->
