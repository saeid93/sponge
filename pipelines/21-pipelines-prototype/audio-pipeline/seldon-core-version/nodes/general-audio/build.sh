REPO=gcr.io/hale-ivy-335012 && \
IMAGE_NAME=audio-pipelines:generalaudio && \
docker build --tag=$IMAGE_NAME . && \
docker tag $IMAGE_NAME $REPO/$IMAGE_NAME && \
docker push $REPO/$IMAGE_NAME