# Deep Q-Learning with Pytorch

The implementation of Deep Q Learning with Pytorch. 

* Replay Memory 
* Simple Deep Q Learning (not using A3C or Dueling) 
* Support for original DQN (the paper in Nature published by DeepMind) and LSTM-based DQN
* Used Pytorch
* Frame Skipping 
* Target Network (for stability when training)
* Python 3.x (I used Python 3.6)

### DQN Algorithm
The linked article explains how DQN works in detail.<br>
[http://andersonjo.github.io/artificial-intelligence/2017/06/03/Deep-Reinforcement-Learning/](http://andersonjo.github.io/artificial-intelligence/2017/06/03/Deep-Reinforcement-Learning/)

### Actual Playing game
The below image is actual result of the code here. 

![alt text](./images/flappybird.gif?raw=true)

[![Watch the video](http://img.youtube.com/vi/MkE6bnK7_DE/0.jpg)](https://youtu.be/MkE6bnK7_DE)

[![Snake Game](https://img.youtube.com/vi/cBxXIII4qRM/0.jpg)](https://www.youtube.com/watch?v=cBxXIII4qRM)


# Installation

Requirements 

1. Python 3
2. Pytorch 
3. TorchVision
4. gym
5. opencv2 
6. Scipy
7. ffmpeg (optional. if you want to record the gameplay)


Install PyGame

```
sudo pip3 install pygame
```

Install PyGame-Learning-Environment

```
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
sudo pip3 install -e .
```

Install Gym-Ple

```
git clone https://github.com/lusob/gym-ple.git
cd gym-ple/
sudo pip3 install -e .
```
## Comparison

| Algorithm | Game | Best Score | 
|:----------|:-----|:-----------|
| DQN       | FlappyBird | 65   |
| LSTM-based DQN | FlappyBird | 83 |

* Best Score is the average value of 10 times of games. 

# How to use

## Training

Before training, you need to make a "dqn_checkpoints" directory for saving model automatically. 

```
mkdir dqn_checkpoints
python3 dqn.py --mode=train
```

Training LSTM-based DQN

```
mkdir dqn_checkpoints
python3 dqn.py --mode=train --model=lstm
```

## Playing 

It automatically loads the latest checkpoint (it loads saved model parameters). <br>
But first, you need to train it.<br>
If there is no checkpoint (You might have not trained it yet), the play is just simply random walk. 

```
python3 dqn.py --mode=play
```

playing LSTM-based DQN is like.. 

```
python3 dqn.py --mode=play --model=lstm
```

## Recoding

If you want to record game play, just do like this. 

```
python3 dqn.py --mode=play --record 
```

## How to convert video to GIF file

```
mkdir frames
ffmpeg -i flappybird.mp4 -qscale:v 2  -r 25 'frames/frame-%05d.jpg'
cd frames
convert -delay 4 -loop 0 *.jpg flappybird.gif
```

FFMpeg and Imagemagic(Convert command) have the following options. 

```
-r 5 stands for FPS value
    for better quality choose bigger number
    adjust the value with the -delay in 2nd step
    to keep the same animation speed

-delay 20 means the time between each frame is 0.2 seconds
   which match 5 fps above.
   When choosing this value
       1 = 100 fps
       2 = 50 fps
       4 = 25 fps
       5 = 20 fps
       10 = 10 fps
       20 = 5 fps
       25 = 4 fps
       50 = 2 fps
       100 = 1 fps
       in general 100/delay = fps

-qscale:v n means a video quality level where n is a number from 1-31, 
   with 1 being highest quality/largest filesize and 
   31 being the lowest quality/smallest filesize.

-loop 0 means repeat forever
```
