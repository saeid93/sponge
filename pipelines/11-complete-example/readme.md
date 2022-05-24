## how to start

- Import by using ```from stress_custom import StressGenerator```
- create instance by 
- ``` sc = StressGenerator(url, is_dataset, datapath, mode)```
- mode is 1 for seldon core, otherwise it have to be 0
- if you want generate load from dataset, dataset should be true, other wise it generates from uniform distribution per seconds.
- use ```sc.set_request_type()``` to post or get request    
- use ``` sc.set_headers``` to set request headers
- use ```sc.set_time(time)``` to set number of seconds of experiment
- use ```sc.run()``` to start experiment