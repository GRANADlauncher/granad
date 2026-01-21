# GRANAD software management

## Overview 
To get the code, do

```
git clone https://github.com/GRANADlauncher/granad/
```

It has two branches:
- *main*: published version, includes
  - time domain simulations for (pulsed) dipole illuminations and plane waves. these can be included in the dipole gauge (x * E) or the velocity gauge (j * A).
  - frequency domain simulations (non-interacting particles and electron repulsion via RPA)
  - plotting utilities and eigenvalue analysis
  - default materials (graphene, chains)
- *dev*: this has all features of main and experimental ones:
  - a self-consistent mean field loop
  - some more materials (ssh, hbn, tmdc)

There is also the branch *gh-pages*, which is exclusively for documentation and doesn't need to be changed directly.

By default, you are on *main* after cloning. To get *dev*, do

```
git switch dev
```

To switch back to main, do

```
git switch main
```

I recommend the following workflow:

- *Always work on dev.*
- *Don't touch main unless people using it report bugs.*
   
## How to change the code

   Change a file, then do

   ```
   git add path/to/file/that/you/changed
   git commit -m "ideally a long and meaningful message where you tell everything you changed"
   git push
   ```

## Documentation
   Both branches have a documentation.

   The main one is:

   https://granadlauncher.github.io/granad/

   The dev one is:

   https://granadlauncher.github.io/granad/dev
   
   The documentation is a collection of html pages. These pages are built from two sources
   
   1. There are markdown files, named like xyz.md, in the folder "granad/docs". Every name corresponds to a page.

       E.g. there is a file "how_to_cite.md". This file becomes the web page https://granadlauncher.github.io/granad/how_to_cite/.

       The file api.md is special. It contains the documentation of all the functions in the code. It contains lines like
     
   ```
   ::: granad.orbitals
   ::: granad.fields
   ```
   
   these lines mean: document all functions in

   ```
   granad/src/granad/orbitals
   granad/src/granad/fields
   ```
   
   2. There is a collection of python files located in the subfolder granad/tutorials. These python files get converted to *jupyter notebooks*. The jupyter notebooks then get converted to markdown files. These get converted to website pages.

      This means: *Every tutorial is an executable jupyter notebook*. For example, you can download the tutorial on linear response as a notebook

      For the *main* version: https://github.com/GRANADlauncher/granad/blob/gh-pages/tutorials/i_linear_response.ipynb

      For the *dev* version: https://github.com/GRANADlauncher/granad/blob/gh-pages/dev/tutorials/i_linear_response.ipynb


   To build the documentation, append the flag "[build-docs]" like this
	   
   ```
   git commit -m "long message [build-docs]"
   ```
   
   For more information, read:
   | Tool                                                                           | Description                                                                                                                                        |
   |--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
   | [mkdocs](https://www.mkdocs.org/)                                                                         | this is a python library. we use it to build the documentation from markdown files. it is configured in granad/mkdocs.yml.                         |
   | [nbconvert](https://nbconvert.readthedocs.io/en/latest/)                                                                       | this is the library we use to convert the python files to jupyter and then to markdown. we do this so mkdocs can turn the md files into web pages. |
   | [github workflows](https://docs.github.com/en/actions/how-tos/write-workflows) | we need to do all of this on github's servers. the commands we give github's servers are in the file "granad/.github/workflows/docs.yml".          |
|                                                                                |                                                                                                                                                    |
   
## New Versions

   If a new version is to be released, we need to *merge* dev into main, i.e. we need to put all the changes in dev into main to update the main branch. This should happen rarely and you can read about this here:

## Tests

   We test in two ways:

   - *git commit -m "[build-docs]"*: The documentation contains the essential features of granad. If something looks weird there or the page doesn't build, this is a failure.
   - *git commit -m "[run-tests]"*: a bunch of small scripts in granad/tests are executed.

     If any of these succeed, there is a little green tick next to the commit message on github. If any of these fail, there is a little red cross.
  
## Scripts

   There is a collection of scripts and utilities here

   https://github.com/granadlauncher/granad-scripts

   These are:
   - experimental features not yet part of GRANAD
   - functionality not covered by GRANAD but used in publications and projects
