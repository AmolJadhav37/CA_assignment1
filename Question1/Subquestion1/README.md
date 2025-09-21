### setup

'''
    $sudo apt install linux-tools-common
'''
This repository contains 4 different variants of the matrix multiplication algorithm programmed by considering different scenario: 

A: with loop order i->j->k sequencial row access
B: with loop order k->i->j 
C: tilted i->j->k loop order 
D: tilted k->i->j loop order 

### obtain perfornamce counter values using following instructions

'''
    make
'''
Then use the below perf tool command find the L1/L2 cache misses, TLB mises, page faults etc.
'''
    perf stat -e cache-references,cache-misses,instructions,cycles,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-icache-load-misses,l2_rqsts.all_demand_miss,l2_rqsts.code_rd_miss,l2_rqsts.demand_data_rd_miss,l2_rqsts.rfo_miss,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses,iTLB-load-misses,page-faults,branches,branch-misses ./Question1_A.out
'''

### in the following command change the ./Question1_A.out change it to ./Question1_B.out ./Question1_C.out ./Question1_D.out to get the performance of all the variants

### this pref command will depend system to system some systems can have more cache levels while some can have few we can get all those hardware performance counters