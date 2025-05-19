# Run from local machin
scp /Users/fabalcu97/.ssh/vastai/github vastai:.ssh/github

# Run on vast.ai machine
ssh-add ~/.ssh/github
eval "$(ssh-agent -s)"
git clone git@github.com:fabalcu97/EmoNeXt.git
git clone git@github.com:fabalcu97/givemefive-dataset.git
cd /workspace/EmoNeXt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

