docker build --tag=stress-pod:latest .
docker tag stress-pod:latest sdghafouri/stress-pod
docker push sdghafouri/stress-pod