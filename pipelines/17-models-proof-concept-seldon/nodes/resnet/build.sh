REPO_NAME=concept-proof
TAG_NAME=resnet
IMAGE_NAME=$REPO_NAME:$TAG_NAME && \
docker build --tag=$IMAGE_NAME . && \
docker tag $IMAGE_NAME sdghafouri/$IMAGE_NAME && \
docker push sdghafouri/$IMAGE_NAME
