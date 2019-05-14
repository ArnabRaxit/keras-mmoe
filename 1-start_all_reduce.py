import sys
print(sys.argv[0].split("-")[0])
print(len(sys.argv))
NUM_WORKERS = len(sys.argv)-1
IP_ADDRS_PORTS = sys.argv[1:]  
[print('%s:%d' % (IP_ADDRS_PORTS[w].split(":")[0], int(IP_ADDRS_PORTS[w].split(":")[1]))) for w in range(NUM_WORKERS)]
