#include <iostream>
#include <vector>
#include <cstring>
#include <limits>
#include <cmath>
#include <cassert>
#include "clustering.h"

struct KVPair
{
    int key;
    double value;
    KVPair()
        :key( 0 )
        ,value( 0.0 )
    {}
    KVPair( int key, double value )
        :key( key)
        ,value( value )
    {}
    KVPair( const KVPair& other )
    {
        key = other.key;
        value = other.value;
    }
};

class Heap 
{
private:
    std::vector<int> index_by_key;
    std::vector<int> key_by_index;
    std::vector<double> values;
    int size;

    static int left_child(int parent) noexcept {
        return (parent << 1) + 1;
    }

    static int parent(int child) noexcept {
        return (child - 1) >> 1;
    }

    void sift_up(int index) noexcept {
        int parent = Heap::parent(index);
        while (index > 0 && values[parent] > values[index]) {
            swap(index, parent);
            index = parent;
            parent = Heap::parent(index);
        }
    }

    void sift_down(int index) noexcept {
        int child = Heap::left_child(index);
        while (child < size) {
            if (child + 1 < size && values[child + 1] < values[child]) {
                child += 1;
            }

            if (values[index] > values[child]) {
                swap(index, child);
                index = child;
                child = Heap::left_child(index);
            } else {
                break;
            }
        }
    }

    void swap(int i, int j) noexcept {
        std::swap(values[i], values[j]);
        int key_i = key_by_index[i];
        int key_j = key_by_index[j];
        key_by_index[i] = key_j;
        key_by_index[j] = key_i;
        index_by_key[key_i] = j;
        index_by_key[key_j] = i;
    }

public:
    Heap(const std::vector<double>& values) {
        size = values.size();
        this->index_by_key.resize(size);
        this->key_by_index.resize(size);
        this->values = values;

        for( int i = 0; i < size; ++i )
        {
            this->index_by_key[i] = i;
            this->key_by_index[i] = i;
        }

        //for (int i = (size - 2) / 2; i >= 0; --i) {
        for (int i = size / 2; i >= 0; --i) {
            sift_down(i);
        }
    }

    KVPair get_min() noexcept {
        return KVPair(key_by_index[0], values[0]);
    }

    void remove_min() noexcept {
        swap(0, size - 1);
        size -= 1;
        sift_down(0);
    }

    void change_value(int key, double value) noexcept {
        int index = index_by_key[key];
        double old_value = values[index];
        values[index] = value;
        if (value < old_value) {
            sift_up(index);
        } else {
            sift_down(index);
        }
    }
};

void get_max_dist_for_each_cluster( const std::vector<std::vector<double>>& Z, 
        std::vector<double>& MD, int n) {
    int k, i_lc, i_rc, root;
    double max_dist, max_l, max_r;
    std::vector<int> curr_node(n);

    // Calculate the size of the 'visited' array
    int visited_size = ((n * 2 - 1) >> 3) + 1;
    unsigned char* visited = new unsigned char[visited_size];
    memset(visited, 0, visited_size);

    k = 0;
    curr_node[0] = 2 * n - 2;
    while (k >= 0) {
        root = curr_node[k] - n;
        i_lc = static_cast<int>(Z[root][0]);
        i_rc = static_cast<int>(Z[root][1]);

        if (i_lc >= n && !(visited[i_lc >> 3] & (1 << (i_lc & 7)))) {
            visited[i_lc >> 3] |= (1 << (i_lc & 7));
            k += 1;
            curr_node[k] = i_lc;
            continue;
        }

        if (i_rc >= n && !(visited[i_rc >> 3] & (1 << (i_rc & 7)))) {
            visited[i_rc >> 3] |= (1 << (i_rc & 7));
            k += 1;
            curr_node[k] = i_rc;
            continue;
        }

        max_dist = Z[root][2];
        if (i_lc >= n) {
            max_l = MD[i_lc - n];
            if (max_l > max_dist) {
                max_dist = max_l;
            }
        }
        if (i_rc >= n) {
            max_r = MD[i_rc - n];
            if (max_r > max_dist) {
                max_dist = max_r;
            }
        }
        MD[root] = max_dist;

        k -= 1;
    }

    delete[] visited;
}

void cluster_monocrit( const std::vector<std::vector<double>>& Z, 
        std::vector<double>& MC, std::vector<int>& T, double cutoff, int n) 
{
    int k, i_lc, i_rc, root, n_cluster = 0, cluster_leader = -1;
    std::vector<int> curr_node(n);

    // Calculate the size of the 'visited' array
    int visited_size = ((n * 2 - 1) >> 3) + 1;
    unsigned char* visited = new unsigned char[visited_size];
    memset(visited, 0, visited_size);

    k = 0;
    curr_node[0] = 2 * n - 2;
    while (k >= 0) {
        root = curr_node[k] - n;
        i_lc = static_cast<int>(Z[root][0]);
        i_rc = static_cast<int>(Z[root][1]);

        if (cluster_leader == -1 && MC[root] <= cutoff) {  // found a cluster
            cluster_leader = root;
            n_cluster += 1;
        }

        if (i_lc >= n && !(visited[i_lc >> 3] & (1 << (i_lc & 7)))) {
            visited[i_lc >> 3] |= (1 << (i_lc & 7));
            k += 1;
            curr_node[k] = i_lc;
            continue;
        }

        if (i_rc >= n && !(visited[i_rc >> 3] & (1 << (i_rc & 7)))) {
            visited[i_rc >> 3] |= (1 << (i_rc & 7));
            k += 1;
            curr_node[k] = i_rc;
            continue;
        }

        if (i_lc < n) {
            if (cluster_leader == -1) {  // singleton cluster
                n_cluster += 1;
            }
            T[i_lc] = n_cluster;
        }

        if (i_rc < n) {
            if (cluster_leader == -1) {  // singleton cluster
                n_cluster += 1;
            }
            T[i_rc] = n_cluster;
        }

        if (cluster_leader == root) {  // back to the leader
            cluster_leader = -1;
        }
        k -= 1;
    }

    delete[] visited;
}

// Calculate the condensed index of element (i, j) in an n x n condensed
// matrix.
int condensed_index(int n, int i, int j) 
{
    if( i < j )
        return n * i - (i * (i + 1) / 2) + (j - i - 1);
    else 
        return n * j - (j * (j + 1) / 2) + (i - j - 1);
}

double _single(double d_xi, double d_yi, double d_xy,
                 int size_x, int size_y, int size_i) 
{
    return std::min(d_xi, d_yi);
}

double _centroid(double d_xi, double d_yi, double d_xy,
                 int size_x, int size_y, int size_i) 
{
    return std::sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                      (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                     (size_x + size_y));
}


KVPair find_min_dist(int n, const std::vector<double>& D, const std::vector<int>& size, int x) {
    double current_min = std::numeric_limits<double>::infinity();
    int y = -1;

    for (int i = x + 1; i < n; ++i) {
        if (size[i] == 0) {
            continue;
        }

        double dist = D[condensed_index(n, x, i)];
        if (dist < current_min) {
            current_min = dist;
            y = i;
        }
    }

    return KVPair(y, current_min);
}

// Only implemented centroid and single distance, here we hard code as centroid 
// if want to add more, just go to source code
// https://github.com/scipy/scipy/blob/main/scipy/cluster/_hierarchy_distance_update.pxi
// then copy code
double new_dist(double d_xi, double d_yi, double d_xy,
                 int size_x, int size_y, int size_i) 
{
    return _centroid(d_xi, d_yi, d_xy, size_x, size_y, size_i);
}


void fast_linkage(std::vector<double>& dists, int n, std::vector<std::vector<double>>& Z) 
{
    //Distances between clusters.
    std::vector<double> D(dists);

    // Sizes of clusters.
    std::vector<int> size(n, 1);
    std::vector<int> cluster_id(n);
    for (int i = 0; i < n; ++i) {
        cluster_id[i] = i;
    }

    // Nearest neighbor candidate and lower bound of the distance to the
    // true nearest neighbor for each cluster among clusters with higher
    // indices (thus size is n - 1).
    std::vector<int> neighbor(n - 1);
    std::vector<double> min_dist(n - 1);
    double dist = 0;
    int id_x = 0;
    int id_y = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;

    int x = 0, y = 0;
    for (x = 0; x < n - 1; ++x) {
        KVPair pair = find_min_dist(n, D, size, x);
        neighbor[x] = pair.key;
        min_dist[x] = pair.value;
    }

    Heap min_dist_heap(min_dist);

    for (int k = 0; k < n - 1; ++k) {
        for (int i = 0; i < n - k; ++i) {
            KVPair pair = min_dist_heap.get_min();
            x = pair.key;
            dist = pair.value;
            y = neighbor[x];

            if (dist == D[condensed_index(n, x, y)]) {
                break;
            }

            pair = find_min_dist(n, D, size, x);
            y = pair.key;
            dist = pair.value;
            neighbor[x] = y;
            min_dist[x] = dist;
            min_dist_heap.change_value(x, dist);
        }
        min_dist_heap.remove_min();

        id_x = cluster_id[x];
        id_y = cluster_id[y];
        nx = size[x];
        ny = size[y];

        if (id_x > id_y) {
            std::swap(id_x, id_y);
        }

        Z[k][0] = id_x;
        Z[k][1] = id_y;
        Z[k][2] = dist;
        Z[k][3] = nx + ny;

        size[x] = 0;
        size[y] = nx + ny;
        cluster_id[y] = n + k;

        // Update the distance matrix.
        for (int z = 0; z < n; ++z) {
            nz = size[z];
            if (nz == 0 || z == y) {
                continue;
            }

            D[condensed_index(n, z, y)] = new_dist(
                D[condensed_index(n, z, x)], D[condensed_index(n, z, y)],
                dist, nx, ny, nz);
        }

        // Reassign neighbor candidates from x to y.
        // This reassignment is just a (logical) guess.
        for (int z = 0; z < x; ++z) {
            if (size[z] > 0 && neighbor[z] == x) {
                neighbor[z] = y;
            }
        }

        // Update lower bounds of distance.
        for (int z = 0; z < y; ++z) {
            if (size[z] == 0) {
                continue;
            }

            dist = D[condensed_index(n, z, y)];
            if (dist < min_dist[z]) {
                neighbor[z] = y;
                min_dist[z] = dist;
                min_dist_heap.change_value(z, dist);
            }
        }

        // Find nearest neighbor for y.
        if (y < n - 1) {
            KVPair pair = find_min_dist(n, D, size, y);
            int z = pair.key;
            dist = pair.value;
            if (z != -1) {
                neighbor[y] = z;
                min_dist[y] = dist;
                min_dist_heap.change_value(y, dist);
            }
        }
    }
}

double euclideanDistance(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double sum = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = static_cast<double>(vec1[i]) - static_cast<double>(vec2[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void Clustering::linkage( const std::vector<std::vector<double>>& input,
        std::vector<std::vector<double>>& dendrogram ) 
{
    size_t n = input.size();
    std::vector<double> distances( n * ( n - 1 ) / 2 );
    int i_index = 0;
    for( size_t i = 0; i < input.size(); ++i )
    {
        for( size_t j = i + 1; j < input.size(); ++j )
        {
            std::vector<double> v1( input[i] );
            std::vector<double> v2( input[j] );
            distances[i_index++] = euclideanDistance( v1, v2 );
        }
    }

    n = input.size();  // Replace with your desired number of clusters
    assert( distances.size() == n * (n - 1) / 2 ); 
    std::vector<std::vector<double>> Z(n - 1, std::vector<double>(4, 0.0));

    fast_linkage(distances, n, Z);

    dendrogram.swap( Z );
}

void Clustering::fcluster( const std::vector<std::vector<double>>& Z, 
        double cutoff, std::vector<int>& clusters ) 
{
    size_t n = Z.size() + 1;

    // Call the get_max_dist_for_each_cluster function
    std::vector<double> MC(n, 0.0);
    get_max_dist_for_each_cluster(Z, MC, n);

    std::vector<int> T(n, 0);

    // Call the cluster_monocrit function
    cluster_monocrit(Z, MC, T, cutoff, n);

    clusters.swap( T );
}

std::vector<int> Clustering::cluster( const std::vector<std::vector<double>>& input, double cutoff ) 
{
    std::vector<std::vector<double>> Z;
    linkage( input, Z);

    std::vector<int> T;
    fcluster(Z, cutoff, T);

    return T;
}

