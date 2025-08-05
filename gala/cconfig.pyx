# cython: language_level=3
# cython: language=c++

if USE_GSL == 1:
    GSL_ENABLED = True
else:
    GSL_ENABLED = False

if USE_EXP == 1:
    EXP_ENABLED = True
else:
    EXP_ENABLED = False
