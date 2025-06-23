# Run from local machine
scp /Users/fabalcu97/.ssh/vastai/github vastai:.ssh/github

# Run on vast.ai machine
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
git clone git@github.com:fabalcu97/EmoNeXt.git
git clone git@github.com:fabalcu97/givemefive-dataset.git
cd /workspace/EmoNeXt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

