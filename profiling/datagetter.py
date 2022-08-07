import sys
def print_var(var_name):
    calling_frame = sys._getframe().f_back
    print(sys._getframe())
    var_val = calling_frame.f_locals.get(var_name, calling_frame.f_globals.get(var_name, None))
    print (var_name+':', str(var_val))

print_var("data")