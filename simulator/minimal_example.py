import gurobipy as gp

model = gp.Model()

x = model.addVar(vtype=gp.GRB.BINARY, name="x")
y = model.addVar(vtype=gp.GRB.BINARY, name="y")

z = model.addVar(vtype=gp.GRB.CONTINUOUS, name="z")

model.update()
constr = model.addConstr((x + y == 2) >> (z <= 20))

model.setObjective(z, gp.GRB.MAXIMIZE)

# Solve the model
model.optimize()

# Print the solution
print(f"x = {x.x}")
print(f"y = {y.x}")
print(f"z = {z.x}")
