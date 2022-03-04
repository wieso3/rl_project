# RL Project: Graph Convolutional Reinforcement Learning

# Installation Guide
NOTE: Work on this was done on a machine running openSUSE Linux. GPU acceleration was used for training baseline policies. Everything else was done on CPU. The enviroments used are only available on Linux! 

Clone this repository first. Using a package manager like anaconda, create a new environment.
```
conda create --name dgn python=3.8
conda activate dgn
```
Install all needed requirements.
```
 pip install -r requirements.txt
```


Experiments can be run, by simply running one of the files **battle.py** or **gather.py**. Which experiment will be executed can be controlled by also providing parameters. Simply run 
```
 python battle.py -h 
```
for displaying help messages.


# What is where?

Training procedures have been implemented for three MAgent environments. More information on these can be found on the official website at https://www.pettingzoo.ml/magent.

It is possible to run (and render) show matches by providing the **-show** option, but atleast for me I could not reproduce trained model behavior consistently, despite saving the entire state dictionary. Due to lack of time I also was only able to provide state dictionaries for battle_v3 with attention.


After that my standard procedure was to do a few training runs. The rewards get dumped to the rewards folder. By running the **utils.py** file, all rewards currently dumped in the rewards folder will be plotted in a graph.
## battle

Here agents compete in two teams. They are rewarded for attacking the enemy team and receive negative rewards when standing around or attacking wrong targets. Access this via:

```
python battle.py -env=battle
```

## battlefield

The setting is the same as battle with a few twists. For a harder challenge there are now fewer agents and additional obstacles on the map. Acces this via:
```
python battle.py -env=battlefield
```

## gather

In this setting there are only blue agents on a map with food (in red). Agents get rewarded for attacking ("eating") the food. They loose help periodically and need to coordinate to make sure everyone survives for a long time, as food restores their health. Access this via:
```
python gather.py
```

### baselines

Baselines can be trained on any of the enviroments. These were mainly used to provide enemy behavior for the battle environments. To train a PPO baseline on any of the enviroments, run the file **train_baseline.py**. Again help is given when providing the *-h* parameter.

# Additional Links

The original paper for MAgent called *'MAgent: A Many-Agent Reinforcement Learning Platform for Artificial Collective Intelligence'* can be found at https://arxiv.org/abs/1712.00600.


Furthermore, the paper on *'Graph Convolutional Reinforcement Learning'* is found at https://arxiv.org/abs/1810.09202.





### Submission Deadline: March. 4th 2022, AOE


