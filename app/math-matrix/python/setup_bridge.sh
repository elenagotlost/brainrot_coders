#!/bin/bash
# Installs the llm-bridge service: lets App Lab containers reach the model
# server by forwarding port 8081 from each Docker network gateway to
# 127.0.0.1:8081. Run once with: sudo bash ~/ArduinoApps/math-matrix/python/setup_bridge.sh
set -e

cat > /usr/local/bin/llm-bridge.sh <<'SCRIPT'
#!/bin/bash
while true; do
  for gw in $(docker network inspect $(docker network ls -q) --format '{{range .IPAM.Config}}{{.Gateway}} {{end}}' 2>/dev/null); do
    case "$gw" in *:*) continue;; esac
    if ! ss -ltn | grep -q "$gw:8081 "; then
      socat TCP-LISTEN:8081,bind=$gw,fork,reuseaddr TCP:127.0.0.1:8081 &
    fi
  done
  sleep 10
done
SCRIPT
chmod +x /usr/local/bin/llm-bridge.sh

cat > /etc/systemd/system/llm-bridge.service <<'UNIT'
[Unit]
Description=Forward the model server to App Lab containers
After=docker.service math-llm.service
Wants=docker.service

[Service]
ExecStart=/usr/local/bin/llm-bridge.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable llm-bridge
systemctl restart llm-bridge
sleep 12
echo "llm-bridge: $(systemctl is-active llm-bridge)"
ss -ltn | grep 8081
