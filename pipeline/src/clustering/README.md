Cluster implementation refer to scipy 

Use chatGpt convert python code to c++, then fix some compile error and debug

Linkage function and its related functions and class in following files:
https://github.com/scipy/scipy/blob/main/scipy/cluster/_hierarchy.pyx
https://github.com/scipy/scipy/blob/main/scipy/cluster/_structures.pxi
https://github.com/scipy/scipy/blob/main/scipy/cluster/_hierarchy_distance_update.pxi


If result is wrong, then compare with original implemention by gdb our c++. But there is problem,
python pdb cannot debug into above pyx and pxi file, which will be compiled as library for scipy.

The solution is download scipy source code, then add log 
print('blabla')
to above source file( pyx and pxi files ).

and compile scipy source code, then run it.

# How to compile scipy
https://scipy.github.io/devdocs/building/index.html

basically create environment with venv module

clone source code and its submodule

$> python dev.py build

$> python dev.py ipython
