

python gen.py

python setup.py clean --all
rm -rf build/ dist/ *.egg-info token_filter/_C*.so

python setup.py build_ext --inplace --define RESTORE_Q 

# --define VERBOSE # --define TORCH25
