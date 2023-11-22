cd ..;

# Create local env:
python3 -m venv mnn_env;
POS=$(pwd);
source $POS/mnn_env/bin/activate

echo $POS/mnn_env/bin/activate

# Install Dependencies 
source $POS/mnn_env/bin/activate && pip install -r $POS/deps/requirements.txt 

# Echo the project alias into the correct rc file 
if [ $SHELL = "/usr/bin/zsh" ]; then 
  echo "alias mnn='cd $POS && source $POS/mnn_env/bin/activate'" >> $HOME/.zshrc;
fi 
if [ $SHELL = "/usr/bin/bash" ]; then 
  echo "alias mnn='cd $POS && source $POS/mnn_env/bin/activate'" >> $HOME/.bashrc;
fi
