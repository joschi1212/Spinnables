This program uses open3d version 0.15.1
Open3d only works with python 3.9 therefore ->

Download Python 3.9 and install
https://www.python.org/downloads/release/python-3912/

check if you really use python 3.9

`$ python --version`

use python 3.9 and install virtualenv with

`$ pip install virtualenv`

open git bash console in root directory
Use python 3.9 and virtualenv to create virtual environment for python inside root directory

`$ python -m virtualenv -p python3.9 venv3.9`

cd into Scripts

`$ cd venv3.9/Scripts/`

activate the virtual environment

`$ source activate`

Console is now in the virtual environment, keep using the same console
using python pip in this console installs every module inside the virtual environment using python 3.9

install all requirements with the requirements.txt in the root directory

`$ cd ../../`

`$ pip install -r requirements`

