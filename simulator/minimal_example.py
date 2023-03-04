import gurobipy as gp

model = gp.Model()

x = model.addVar(vtype=gp.GRB.BINARY, name="x")
y = model.addVar(vtype=gp.GRB.BINARY, name="y")
a = model.addVar(vtype=gp.GRB.BINARY, name="a")


z = model.addVar(vtype=gp.GRB.CONTINUOUS, name="z", ub=30)
model.addGenConstrAnd(a, [x, y], "andconstr")

model.update()

# constr1 = model.addConstr(z <= 20)
constr2 = model.addConstr((a == 1) >> (z <= 5))

model.setObjective(z+x+y, gp.GRB.MAXIMIZE)

model.update()
# Solve the model
model.optimize()
model.display()