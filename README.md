# Active Preference Learning with Discrete Choice Data
This is an unofficial implementation of the [Active Preference Learning with Discrete Choice Data](https://papers.nips.cc/paper/2007/hash/b6a1085a27ab7bff7550f8a3bd017df8-Abstract.html) by Brochu et al. as published in NIPS 2007.

Why would this package be useful for you?
> Imagine a scenario where you are trying to find a place to have lunch today. 
> There are tons of places to eat around. 
> An app presents you two restaurants to compare at a time and can help you reach a good enough restaurant in as few queries as possible.
> Each time you pick a restaurant, the model gradually learns what you want and hopefully, its suggestions get gradually better.
> This will save you a lot of time and when designed well, can be a much more fun way to search as opposed to going through a boring list view.

![](readme_photo.jpg)

In general, if the following conditions are present, active preference can be useful for you:
- User is searching for an item in a very large set of items that's impossible go through one by one.
- User is okay with a good enough solution if it is going to be found shortly.
- Items can be embedded in a vectors space where proximity in that space implies similarity in preference between items.

# Installation
There are currently two modes of installation: bare bones, extras, development.

Whichever mode, first clone the repository.

`git clone git@github.com:dorukhansergin/APL-Brochu.git`

## Bare Bones
The package requires:
```
numpy~=1.20.1 
scikit-learn~=0.24.1 
scipy
```

Change into the folder of the cloned repository and use `pip` to install.

`pip install .`

## Extras
In addition to bare bones, the following packages will be installed:
```
streamlit
matplotlib
```

`pip install .\[extras\]`

## Development
In addition to bare bones, the following packages will be installed:
```
pytest
pylint 
black 
rope
```

Change into the folder of the cloned repository and use `pip` to install, with the dev mode. The `-e` flag listens to change in code so it's recommended.

`pip install -e .\[dev\]`

You can run the tests using pytest with the simple `pytest` command.

# Play with the Demo
See above for the installation of extras.
The extras has a demo for you to get a feeling for the algorithm.
Use the streamlit command to run it.

`streamlit run extras/brochu2d.py`

