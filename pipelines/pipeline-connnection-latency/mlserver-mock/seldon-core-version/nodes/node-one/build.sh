REPOS=(
    sdghafouri)
IMAGE_NAME=grpc-pipeline:node-one
# mlserver build . -t $IMAGE_NAME
DOCKER_BUILDKIT=1 docker build . --tag=$IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done