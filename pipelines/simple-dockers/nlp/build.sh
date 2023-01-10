REPOS=(
    sdghafouri)
IMAGE_NAME=simple-image:nlp
DOCKER_BUILDKIT=1 docker build . --tag=$IMAGE_NAME
for REPO in ${REPOS[@]}
do
    docker tag $IMAGE_NAME $REPO/$IMAGE_NAME
    docker push $REPO/$IMAGE_NAME
done