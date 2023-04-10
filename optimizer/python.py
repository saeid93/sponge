import gurobipy as gp

# Define the set of values
values = [1, 3, 5, 7]

# Create a new model
model = gp.Model()

# Create binary indicator variables
indicators = {}
for val in values:
    indicators[val] = model.addVar(vtype=gp.GRB.BINARY)

# Create the integer variable to be selected from the set
selected = model.addVar(vtype=gp.GRB.INTEGER, lb=min(values), ub=max(values))

# Add constraints to ensure that only one value is selected
model.addConstr(gp.quicksum(indicators[val] for val in values) == 1)

# Add constraints to enforce the indicator variables
for val in values:
    model.addConstr(selected >= val - (max(values) - min(values)) * (1 - indicators[val]))
    model.addConstr(selected <= val + (max(values) - min(values)) * (1 - indicators[val]))

# Set the objective function
model.setObjective(-1 * selected, gp.GRB.MINIMIZE)

# Optimize the model
model.optimize()

# Print the solution
for val in values:
    print(val, '=', indicators[val].x)
print('selected =', selected.x)
