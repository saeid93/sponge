import time
import multiprocessing

def function1():
    # code for function 1
    counter = 0
    while counter != 20:
        print(f'func 1 counter {counter}')
        time.sleep(2)
        counter += 1

def function2():
    # code for function 2
    counter = 0
    while counter != 30:
        print(f'func 2 counter {counter}')
        time.sleep(3)
        counter += 1

# create processes
p1 = multiprocessing.Process(target=function1)
p2 = multiprocessing.Process(target=function2)

# start processes
p1.start()
p2.start()

# wait for processes to finish
p1.join()
p2.join()