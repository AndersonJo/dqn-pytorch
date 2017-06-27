# Deep Q-Learning with Pytorch

The implementation of Deep Q Learning with Pytorch. 

* Replay Memory 
* Simple Deep Q Learning (not using A3C or Dueling) 
* Used Pytorch
* Frame Skipping 
* Target Network (for stability when training)
* Python 3.x (I used Python 3.6)

Here you can see the explanation of DQN algorithm. <br>
[http://andersonjo.github.io/artificial-intelligence/2017/06/03/Deep-Reinforcement-Learning/](http://andersonjo.github.io/artificial-intelligence/2017/06/03/Deep-Reinforcement-Learning/)

The below image is actual result of the code here. 

![alt text](./images/flappybird.gif?raw=true)


# Installation

of course, You need Pytorch but I am not going to tell how to install Pytorch here. 

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

# How to use

## Training

```
python3 dqn.py --mode=train
```

## Playing 

```
python3 dqn.py --mode=play
```

## Recoding

```
python3 dqn.py --mode=play --record 
```

## How to convert video to GIF file

```
mkdir frames
ffmpeg -i video.mp4 -qscale:v 2  -r 25 'frames/frame-%03d.jpg'
cd frames
convert -delay 4 -loop 0 *.jpg myimage.gif
```
