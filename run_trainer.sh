python trainer.py \
     --ps_hosts=192.168.8.106:2222,192.168.8.103:2222 \
     --worker_hosts=192.168.8.106:2223,192.168.8.103:2223 \
     --job_name=ps --task_index=0


python trainer.py \
     --ps_hosts=192.168.8.106:2222,192.168.8.103:2222 \
     --worker_hosts=192.168.8.106:2223,192.168.8.103:2223 \
     --job_name=ps --task_index=1


python trainer.py \
     --ps_hosts=192.168.8.106:2222,192.168.8.103:2222 \
     --worker_hosts=192.168.8.106:2223,192.168.8.103:2223 \
     --job_name=worker --task_index=0


python trainer.py \
     --ps_hosts=192.168.8.106:2222,192.168.8.103:2222 \
     --worker_hosts=192.168.8.106:2223,192.168.8.103:2223 \
     --job_name=worker --task_index=1

