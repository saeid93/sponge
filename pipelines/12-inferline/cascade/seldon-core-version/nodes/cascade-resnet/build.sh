docker build --tag=inferline:cascade-resnet . && \
docker tag linearmodel:nodeone sdghafouri/inferline:cascade-resnet && \
docker push sdghafouri/inferline:cascade-resnet
