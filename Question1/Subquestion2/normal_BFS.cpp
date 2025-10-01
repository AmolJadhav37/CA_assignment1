#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <bits/stdc++.h>

using namespace std;

// Simple graph structure using adjacency list
struct Graph {
    int n;  // number of vertices
    vector<vector<int>> adj;  // adjacency list
    
    Graph(int vertices) : n(vertices), adj(vertices) {}
    
    void add_edge(int u, int v) {
        adj[u].push_back(v);
    }
};

// Generate Kronecker graph (similar to GAP benchmark -g parameter)
Graph generate_kronecker_graph(int scale) {
    int n = 1 << scale;  // 2^scale vertices
    Graph g(n);
    
    // Kronecker graph parameters (similar to GAP)
    double a = 0.57, b = 0.19, c = 0.19;  // d = 1-(a+b+c) = 0.05
    
    mt19937 rng(12345);  // Fixed seed for reproducibility
    uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Generate approximately 16*n edges (GAP uses average degree of 16)
    long long num_edges = 16LL * n;
    
    for (long long e = 0; e < num_edges; e++) {
        int u = 0, v = 0;
        
        for (int level = 0; level < scale; level++) {
            double r = dist(rng);
            int u_bit, v_bit;
            
            if (r < a) {
                u_bit = 0; v_bit = 0;
            } else if (r < a + b) {
                u_bit = 0; v_bit = 1;
            } else if (r < a + b + c) {
                u_bit = 1; v_bit = 0;
            } else {
                u_bit = 1; v_bit = 1;
            }
            
            u |= (u_bit << level);
            v |= (v_bit << level);
        }
        
        if (u != v) {  // No self-loops
            g.add_edge(u, v);
        }
    }
    
    return g;
}

// Simple unoptimized BFS - returns distance array
vector<int> bfs(const Graph& g, int source) {
    vector<int> dist(g.n, -1);
    queue<int> q;
    
    // Start BFS from source
    dist[source] = 0;
    q.push(source);
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        // Explore all neighbors
        for (int v : g.adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    
    return dist;
}



void print_usage(const char* prog) {
    cout << "Usage: " << prog << " -g <scale> -n <num_trials>" << endl;
    cout << "  -g: Graph scale (generates 2^scale vertices)" << endl;
    cout << "  -n: Number of trials to run" << endl;
    cout << "Example: " << prog << " -g 25 -n 1" << endl;
}

int main(int argc, char* argv[]) {
    int scale = -1;
    int num_trials = 1;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            scale = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_trials = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (scale == -1) {
        print_usage(argv[0]);
        return 1;
    }
    
    cout << "Generating Kronecker graph with scale " << scale << "..." << endl;
    auto gen_start = chrono::high_resolution_clock::now();
    Graph g = generate_kronecker_graph(scale);
    auto gen_end = chrono::high_resolution_clock::now();
    
    long long total_edges = 0;
    for (int i = 0; i < g.n; i++) {
        total_edges += g.adj[i].size();
    }
    
    double gen_time = chrono::duration<double>(gen_end - gen_start).count();
    cout << "Graph generated: " << g.n << " vertices, " << total_edges << " edges" << endl;
    cout << "Generation time: " << gen_time << " seconds" << endl;
    cout << endl;
    
    // Run BFS trials
    int source = 0;  // Start from vertex 0
    vector<double> trial_times;
    
    for (int trial = 0; trial < num_trials; trial++) {
        cout << "Trial " << (trial + 1) << "..." << endl;
        
        auto bfs_start = chrono::high_resolution_clock::now();
        vector<int> dist = bfs(g, source);
        auto bfs_end = chrono::high_resolution_clock::now();
        
        double bfs_time = chrono::duration<double>(bfs_end - bfs_start).count();
        trial_times.push_back(bfs_time);
        
        cout << "  BFS time: " << bfs_time << " seconds" << endl;
        
        // Count reachable vertices
        int reachable = 0;
        for (int i = 0; i < g.n; i++) {
            if (dist[i] != -1) reachable++;
        }
        cout << "  Reachable vertices: " << reachable << endl;
        
        // Verify result
        bool valid = verify_bfs(g, source, dist);
        cout << "  Verification: " << (valid ? "PASSED" : "FAILED") << endl;
        cout << endl;
    }
    
    // Print summary
    if (num_trials > 1) {
        double avg_time = 0;
        for (double t : trial_times) avg_time += t;
        avg_time /= num_trials;
        
        cout << "Average BFS time: " << avg_time << " seconds" << endl;
    }
    
    return 0;
}