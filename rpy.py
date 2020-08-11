# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:49:49 2020

@author: Propietario
"""
import os
import math, datetime
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

base = importr('base')