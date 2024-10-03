TAG=$(git rev-parse --short HEAD)

docker build --platform linux/amd64 --tag astrohyo/suprem-runpod-worker:${TAG} .
docker push astrohyo/suprem-runpod-worker:${TAG}