p= ps aux | grep "ai21m034/.cache/JetBrains/RemoteDev"  | awk '{print $2}'
for k in $p; do kill $k; done
