# Grimme D3-dispersion energy and their derivatives

- This is an temporary experimental side-repo derivative work of this work: https://github.com/Rafael-G-C/pyDFTD3
- Which is a derivative work of https://github.com/bobbypaton/pyDFTD3, (c) 2020 Rob Paton and contributors
- Which is a Python reimplementation of the Grimme D3-dispersion energy


## Notes about this repository

- The code can and probably will be sent as pull request towards
  https://github.com/Rafael-G-C/pyDFTD3 but I didn't want to undo a lot of
  other people's work with one PR without asking so this requires some
  discussion first.
- For simplicity when refactoring I have removed the Becke-Johnson damping
  scheme but it can be reintroduced.
- I have moved out all parameter and I/O out of the d3 function so that it does
  not have to know anything about functionals and setting the appropriate
  parameters can be done caller-side.
- It turned out to be significantly more efficient to compute all derivatives
  of one order in one shot. [demo.py](demo.py) shows how this can be done.
- Distance-based screening can be done but is not completely trivial and hasn't
  been attempted in this rewrite.
