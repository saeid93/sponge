import gurobipy as gp
from gurobipy import GRB

def solve_cubic_optimization():
    # Create a Gurobi model
    model = gp.Model("CubicOptimization")

    # Define decision variables
    x = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

    # Define the cubic objective function
    objective_expr = x**3 - 4*x**2 + 2*x + 1
    model.setObjective(objective_expr, GRB.MINIMIZE)

    # Add constraints (optional)
    # You can add constraints to the model if needed
    # For example: model.addConstr(x >= 0)

    # Optimize the model
    model.optimize()

    # Print the optimal solution and objective value
    if model.status == GRB.OPTIMAL:
        print(f"Optimal value of x: {x.x}")
        print(f"Optimal objective value: {model.objVal}")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    solve_cubic_optimization()
