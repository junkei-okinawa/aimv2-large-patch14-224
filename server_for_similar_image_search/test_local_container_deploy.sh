docker build -f ./Dockerfile -t image_searcher .
docker run -p 8000:8080 -e PORT=8080 -itd image_searcher
sleep 5
cloudflared tunnel --url 127.0.0.1:8000
