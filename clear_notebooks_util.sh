#based off of https://mg.readthedocs.io/git-jupyter.html#cleaning-a-whole-repository
git filter-branch --tree-filter "find . -name "*.ipynb" -exec python3 -m nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} || true"
