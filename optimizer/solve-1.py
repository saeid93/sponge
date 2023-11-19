import gurobipy as gp
from gurobipy import GRB

# Create a model
model = gp.Model("SimpleLP")

# Add decision variables
x1 = model.addVar(name="x1", vtype=GRB.BINARY, lb=0)
# x2 = model.addVar(name="x2", vtype=GRB.CONTINUOUS, lb=0)
# x3 = model.addVar(name="x3", vtype=GRB.CONTINUOUS, lb=0)
# z = model.addVar(name="z", vtype=GRB.CONTINUOUS, lb=0, ub=1000)


# Add constraints
# 2x1 + x2 <= 10
model.addConstr(x1 == 1, name="c1")
# 4x1 - 5x2 >= -8
# model.addConstr(4 * x1 - 5 * x2 >= -8, name="c2")
# # x1 + 2x2 = 7
# model.addConstr(x1 + 2 * x2 == 7, name="c3")

# model.addConstr(x1 + 2 * x3 == 8, name="c4")

# model.addConstr(z == 3 * x3, name="c5")


# Set objective function: Maximize 3x1 + 2x2
model.setObjective(x1, sense=GRB.MAXIMIZE)

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("\nOptimal solution:")
    print(f"x1 = {x1.x}")
    # print(f"x2 = {x2.x}")
    # print(f"x3 = {x3.x}")
    print(f"Objective value = {model.objVal}")
else:
    print("Optimization did not converge")

# Save the model to a file (optional)
model.write("simple_lp_model.lp")
