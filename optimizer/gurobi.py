# Import packages
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Sets P and D, respectively
# When we code sets we can be more descriptive in the name
production = ['Baltimore','Cleveland','Little Rock','Birmingham','Charleston']
distribution = ['Columbia','Indianapolis','Lexington','Nashville','Richmond','St. Louis']

# Define a gurobipy model for the decision problem
m = gp.Model('widgets')

# Use squeeze=True to make the costs a series
path = 'https://raw.githubusercontent.com/Gurobi/modeling-examples/master/optimization101/Modeling_Session_1/'
transp_cost = pd.read_csv(path + 'cost.csv', index_col=[0,1], squeeze=True)

# Pivot to view the costs a bit easier
transp_cost.reset_index().pivot(index='production', columns='distribution', values='cost')

max_prod = pd.Series([180,200,140,80,180], index = production, name = "max_production")
n_demand = pd.Series([89,95,121,101,116,181], index = distribution, name = "demand") 
max_prod.to_frame()
#n_demand.to_frame()