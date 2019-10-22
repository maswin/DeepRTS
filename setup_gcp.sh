# Setup Ubuntu
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git

# Get the project
git clone https://github.com/maswin/DeepRTS.git
cd DeepRTS/

# Setup python
sudo apt-get install python3.7
sudo apt-get install python3-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py

# Setup cmake
sudo apt-get -y install cmake
sudo apt-get install -y build-essential

# Setup dependencies
git submodule sync
git submodule update --init
sudo apt-get install python3-dev

# Install package
sudo pip install .

# Setup model packages
cd python
sudo pip install -r requirements.txt

# Install tmux
sudo apt-get install tmux
tmux new -s deep
# ctrl+b d = to detach
tmux a -t deep
# ctrl-b % : Horizontal split
# ctrl-b " : Vertical split

# Run code
PYTHONPATH=$PYTHONPATH:. python3 DeepRTS/run.py 



