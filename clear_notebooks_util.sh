#based off of https://mg.readthedocs.io/git-jupyter.html#cleaning-a-whole-repository
echo "are you sure you want to do this? if so uncomment the line below"
#git filter-branch -f --tree-filter 'find . -name "*.ipynb" -exec python3 -m nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \; || true'
