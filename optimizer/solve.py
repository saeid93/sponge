import gurobipy as gp
from gurobipy import GRB

# Create a model
model = gp.Model("SimpleLP")

# Add decision variables
x1 = model.addVar(name="x1", vtype=GRB.INTEGER, lb=0, ub=1)
# x2 = model.addVar(name="x2", vtype=GRB.CONTINUOUS, lb=0, ub=5)
z = model.addVar(name="z", vtype=GRB.CONTINUOUS, lb=0)
x11 = model.addVar(name="x11", vtype=GRB.BINARY, lb=0, ub=1)
x12 = model.addVar(name="x12", vtype=GRB.BINARY, lb=0, ub=1)
x13 = model.addVar(name="x13", vtype=GRB.BINARY, lb=0, ub=1)
# Set objective function: Maximize 3x1 + 2x2
# Add constraints
# 2x1 + x2 <= 10
# model.addConstr(2 * x1 + x2 <= 10, name="c1")
# 4x1 - 5x2 >= -8
# model.addConstr(4 * x1 - 5 * x2 >= -8, name="c2")
# x1 + 2x2 = 7
# model.addConstr(x1 + 2 * x2 == 7, name="c3")

# model.addConstr(x1 + 2 * x3 == 8, name="c4")
model.addConstr(x1 == 1)
model.addConstr(x1 == (x11 + x12 + x13) / 3)
# model.addConstr(x2 == 1)
model.addConstr((x1 == 1) >> (x1 == 2 * z))

model.setObjective(x1 + z, sense=GRB.MAXIMIZE)

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("\nOptimal solution:")
    print(f"x1 = {x1.x}")
    # print(f"x2 = {x2.x}")
    print(f"z = {z.x}")
    print(f"x1 = {x11.x}")
    print(f"x1 = {x12.x}")
    print(f"x1 = {x13.x}")
    print(f"Objective value = {model.objVal}")
else:
    print("Optimization did not converge")

# Save the model to a file (optional)
model.write("simple_lp_model.lp")
