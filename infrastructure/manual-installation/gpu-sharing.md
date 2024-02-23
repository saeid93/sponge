# GPU sharing
In this md file, steps to enable gpu-sharing in micorok8s is shared.
## Prerequisite
check if your system has Nvidia gpus and their drivers are installed.
```
nvidia-smi
```
## Enabling GPU sharing
 1. enable GPU

   Use the following command to enable gpu.
   
  	```
  	microk8s enable gpu
  	```
 3. create GPU time-slicing config
	 
	```
	cat << EOF > time-slicing-config-all.yaml
	apiVersion: v1
	kind: ConfigMap
	metadata:
	  name: time-slicing-config-all
	data:
	  any: |-
	    version: v1
	    flags:
	      migStrategy: none
	    sharing:
	      timeSlicing:
	        resources:
	        - name: nvidia.com/gpu
	          replicas: 4
	EOF
	```
4. Add the config map to the same namespace as the GPU operator
	```
	kubectl create -n gpu-operator-resources -f time-slicing-config-all.yaml
	```	
5. Configure the device plugin with the config map and set the default time-slicing configuration:
	```
	kubectl patch clusterpolicy/cluster-policy \
	    -n gpu-operator-resources --type merge \
	    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config-all", "default": "any"}}}}'
	```
Now you are all set.
Happy codding!