while true
    sleep 0.1
    do
    curl -s -d '{"data": {"ndarray":[[1.0, 2.0, 5.0, 6.0]]}}' \
    -X POST http://localhost:32000/seldon/seldon/$1/api/v1.0/predictions \
    -H "Content-Type: application/json"
    done
