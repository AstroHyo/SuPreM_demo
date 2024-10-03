TAG=$(git rev-parse --short HEAD)

docker build --tag astrohyo/suprem-runpod-worker:${TAG} .
docker push astrohyo/suprem-runpod-worker:${TAG}