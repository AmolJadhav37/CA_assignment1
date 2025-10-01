### clone the gaps repository using following command 

```bash
    git clone https://github.com/sbeamer/gapbs.git
```

### build the project using built in makefile

```bash
    make
```

### run BFS on 2^25 vertices vertices for 1 iteration

```bash
    ./bfs -g 25 -n 1
```
here 25 reperesents the 2^25 vertices and 1 represent the number of interation 

### run the performance stats using pref tool on the bfs 
```bash
perf stat \
  -e cpu-cycles \
  -e instructions \
  -e cache-references \
  -e cache-misses \
  -e L1-dcache-loads \
  -e L1-dcache-load-misses \
  -e L1-dcache-stores \
  -e L1-dcache-store-misses \
  -e l2_rqsts.demand_data_rd_hit \
  -e l2_rqsts.demand_data_rd_miss \
  -e LLC-loads \
  -e LLC-load-misses \
  -e LLC-stores \
  -e LLC-store-misses \
  -e dTLB-loads \
  -e dTLB-load-misses \
  -e branch-loads \
  -e branch-load-misses \
    ./bfs -g 25 -n 1
```

### standard roofline intensity:

arithmatic intensity = FLOPs / Bytes accessed from DRAM 

### intensity for BFS 

BFS don't have floating point operations so we are considering edges traversed per second as total work done. 

hence , Intensity = Edges Traversed / Bytes accessed from DRAM 

for the roofline graphs of both the benchmakrs we used google collab and for python code generation we used ClAUDE.ai and contructed the graphs for the analysis 

to write the code of normal BFS we took the help of AI it is not completly done by ourself 