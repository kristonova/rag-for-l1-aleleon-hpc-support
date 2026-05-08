# Promtail's container scraping

## Setup rootless Podman socket
1. Stop all containers
3. Remove Podman socket's XDG dir: `rm /run/user/1000/podman/podman.sock -rf`
4. Enable and start rootless Podman socket: `systemctl --user enable --now podman.socket`


## Run the Promtail logging
`podman-compose --profile monitoring up -d`

## How to check for logs
[Grafana Logs](http://172.16.1.10:3000/a/grafana-lokiexplore-app/explore)