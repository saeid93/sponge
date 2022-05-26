IMAGE_NAME=inferline:cascade-resnet && \
docker build --tag=$IMAGE_NAME . && \
docker tag $IMAGE_NAME sdghafouri/$IMAGE_NAME && \
docker push sdghafouri/$IMAGE_NAME
