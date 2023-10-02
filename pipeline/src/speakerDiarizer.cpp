// Copyright (c) 2023 Huang Liyi (webmaster@360converter.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <chrono>
#include <sstream>

#include "frontend/wav.h"
#include "clustering/clustering.h"

//#include "kmeans.h"


//#include <torch/torch.h>
#include <torch/script.h>

#include "onnxModel/onnx_model.h"

#define SAMPLE_RATE 16000

//#define WRITE_DATA 1

// python: min_num_samples = self._embedding.min_num_samples
size_t min_num_samples = 640;
int g_sample_rate = 16000;


std::chrono::time_point<std::chrono::high_resolution_clock> timeNow()
{
    return std::chrono::high_resolution_clock::now();
}

void timeCost( std::chrono::time_point<std::chrono::high_resolution_clock> beg,
        std::string label )
{
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
    std::cout<<"-----------"<<std::endl;
    std::cout <<label<<": "<< duration.count()<<"ms"<<std::endl;
}

template<class T>
void debugPrint2D( const std::vector<std::vector<T>>& data, std::string dataInfo )
{
    std::cout<<"--- "<<dataInfo<<" ---"<<std::endl;
    for( const auto& a : data )
    {
        for( T b : a )
        {
            std::cout<<b<<",";
        }
        std::cout<<std::endl;
    }
}

template<class T>
void debugPrint( const std::vector<T>& data, std::string dataInfo )
{
    std::cout<<"--- "<<dataInfo<<" ---"<<std::endl;
    for( T b : data )
    {
        std::cout<<b<<",";
    }
    std::cout<<std::endl;
}

template<class T>
void debugWrite( const std::vector<T>& data, std::string name, bool writeDecimalforZero = false )
{
    std::string fileName = "/tmp/";
    fileName += name;
    fileName += ".txt";
    std::ofstream f( fileName );
    for( T b : data )
    {
        if( std::is_same<T, bool>::value )
        {
            std::string tmp = b ? "True" : "False";
            f<<tmp<<",";
        }
        else if( std::is_same<T, float>::value )
        {
            if( writeDecimalforZero )
            {
                std::ostringstream oss;
                oss << std::setprecision(1) << b;
                std::string result = oss.str();
                result = result == "1" ? "1.0" : result;
                result = result == "0" ? "0.0" : result;
                f<<result<<",";
            }
            else
            {
                f<<b<<",";
            }
        }
        else
        {
            f<<b<<",";
        }
    }
    f.close();
}

/*
 * writeDecimalforZero: if enabled, write 0 as 0.0
 * */
template<class T>
void debugWrite2d( const std::vector<std::vector<T>>& data, std::string name, bool writeDecimalforZero = false )
{
    std::string fileName = "/tmp/";
    fileName += name;
    fileName += ".txt";
    std::ofstream f( fileName );
    for( const auto& a : data )
    {
        for( T b : a )
        {
            if( std::is_same<T, bool>::value )
            {
                std::string tmp = b ? "True" : "False";
                f<<tmp<<",";
            }
            else if( std::is_same<T, float>::value )
            {
                if (std::isnan( b )) 
                {
                    f<<"nan,";
                }
                else
                {
                    if( writeDecimalforZero )
                    {
                        std::ostringstream oss;
                        oss << std::setprecision(6) << b;
                        std::string result = oss.str();
                        result = result == "1" ? "1.0" : result;
                        result = result == "0" ? "0.0" : result;
                        f<<result<<",";
                    }
                    else
                    {
                        f<<b<<",";
                    }
                }
            }
            else
            {
                f<<b<<",";
            }
        }
        f<<"\n";
    }
    f.close();
}

/*
 * writeDecimalforZero: if enabled, write 0 as 0.0
 * haveNan: if has Nan, which will be written as 'nan'
 * later easier to compare with python result or load into python with loadtxt()
 * */
template<class T>
void debugWrite3d( const std::vector<std::vector<std::vector<T>>>& data, std::string name, 
        bool writeDecimalforZero = false )
{
    std::string fileName = "/tmp/";
    fileName += name;
    fileName += ".txt";
    std::ofstream f( fileName );
    for( const auto& a : data )
    {
        for( const auto& b : a )
        {
            for( T c : b )
            {
                if( std::is_same<T, bool>::value )
                {
                    std::string tmp = c ? "True" : "False";
                    f<<tmp<<",";
                }
                else if( std::is_same<T, float>::value ||
                    std::is_same<T, double>::value )
                {
                    if (std::isnan( c )) 
                    {
                        f<<"nan,";
                    }
                    else
                    {
                        if( writeDecimalforZero )
                        {
                            std::ostringstream oss;
                            oss << std::setprecision(6) << c;
                            std::string result = oss.str();
                            result = result == "1" ? "1.0" : result;
                            result = result == "0" ? "0.0" : result;
                            f<<result<<",";
                        }
                        else
                        {
                            f<<c<<",";
                        }
                    }
                }
                else
                {
                    f<<c<<",";
                }
            }
            f<<"\n";
        }
    }
    f.close();
}

class Helper 
{
public:

    // for string delimiter
    static std::vector<std::string> split(std::string s, std::string delimiter) 
    {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
            token = s.substr (pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back (token);
        }

        res.push_back (s.substr (pos_start));
        return res;
    }

    // Mimic python np.rint
    // For values exactly halfway between rounded decimal values, 
    // NumPy rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc.
    template <typename T>
    static int np_rint( T val )
    {
        if(abs( val - int( val ) - 0.5 * ( val > 0 ? 1 : -1 )) < std::numeric_limits<double>::epsilon())
        {
            int tmp = std::round( val );
            if( tmp % 2 == 0 )
                return tmp;
            else
                return tmp - 1 * ( val > 0 ? 1 : -1 );
        }
        return std::round( val );
    }

    template <typename T>
    static std::vector<int> argsort(const std::vector<T> &v) 
    {

        // initialize original index locations
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(idx.begin(), idx.end(),
                [&v](int i1, int i2) {return v[i1] < v[i2];});

        return idx;
    }

    // python: hard_clusters = np.argmax(soft_clusters, axis=2)
    template <typename T>
    static std::vector<std::vector<int>> argmax( std::vector<std::vector<std::vector<T>>>& data )
    {
        std::vector<std::vector<int>> res( data.size(), std::vector<int>( data[0].size()));
        for( size_t i = 0; i < data.size(); ++i )
        {
            for( size_t j = 0; j < data[0].size(); ++j )
            {
                int max_index = 0;
                double max_value = -1.0 * std::numeric_limits<double>::max();
                for( size_t k = 0; k < data[0][0].size(); ++k )
                {
                    if( data[i][j][k] > max_value )
                    {
                        max_index = k;
                        max_value = data[i][j][k];
                    }
                }
                res[i][j] = max_index;
            }
        }

        return res;
    }

    // Define a helper function to find non-zero indices in a vector
    static std::vector<int> nonzeroIndices(const std::vector<bool>& input) 
    {
        std::vector<int> indices;
        for (int i = 0; i < input.size(); ++i) {
            if (input[i]) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    // Function to compute the L2 norm of a vector
    template <typename T>
    static float L2Norm(const std::vector<T>& vec) 
    {
        T sum = 0.0;
        for (T val : vec) 
        {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    // Function to normalize a 2D vector
    template <typename T>
    static void normalizeEmbeddings(std::vector<std::vector<T>>& embeddings) 
    {
        for (std::vector<T>& row : embeddings) 
        {
            T norm = L2Norm(row);
            if (norm != 0.0) 
            {
                for (T& val : row) 
                {
                    val /= norm;
                }
            }
        }
    }

    // Function to calculate the Euclidean distance between two vectors
    template <typename T>
    static T euclideanDistance(const std::vector<T>& vec1, const std::vector<T>& vec2) {
        T sum = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            T diff = static_cast<T>(vec1[i]) - static_cast<T>(vec2[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // scipy.cluster.hierarchy.linkage with method: single
    template <typename T>
    static T clusterDistance_single( const std::vector<std::vector<T>>& embeddings,
            const std::vector<std::vector<T>>& distances,
            const std::vector<int>& cluster1, 
            const std::vector<int>& cluster2 )
    {
        T minDistance = 1e9;
        for( size_t i = 0; i < cluster1.size(); ++i )
        {
            if( cluster1[i] == -1 )
                return minDistance;
            // calc each cluster distance
            for( size_t j = 0; j < cluster2.size(); ++j )
            {
                if( cluster2[j] == -1 )
                    break;
                int _i = cluster1[i];
                int _j = cluster2[j];
                if( _i > _j )
                {
                    _i = cluster2[j];
                    _j = cluster1[i];
                }
                assert( _i != _j ); // same point cannot be in cluster1 and cluster2 at same time
                T dis = distances[_i][_j];
                if( dis < minDistance )
                    minDistance = dis;
            }
        }

        return minDistance;
    }
            
    // scipy.cluster.hierarchy.linkage with method: centroid
    template <typename T>
    static T clusterDistance_centroid( const std::vector<std::vector<T>>& embeddings,
            const std::vector<std::vector<T>>& distances,
            const std::vector<int>& cluster1, 
            const std::vector<int>& cluster2 )
    {
        T minDistance = 1e9;
        // d(i ∪ j, k) = αid(i, k) + αjd(j, k) + βd(i, j)
        // αi = |i| / ( |i|+|j| ), αj = |j| / ( |i|+|j| )
        // β = − |i||j| / (|i|+|j|)^2
        for( size_t i = 0; i < cluster1.size(); ++i )
        {
            if( cluster1[i] == -1 )
                return minDistance;
            // calc each cluster distance
            for( size_t j = 0; j < cluster2.size(); ++j )
            {
                if( cluster2[j] == -1 )
                    return minDistance;
                int _i = cluster1[i];
                int _j = cluster2[j];
                if( _i > _j )
                {
                    _i = cluster2[j];
                    _j = cluster1[i];
                }
                assert( _i != _j ); // same point cannot be in cluster1 and cluster2 at same time
                T dis = distances[_i][_j];
                if( dis < minDistance )
                    minDistance = dis;
            }
        }

        return minDistance;
    }

    // Function to calculate the mean of embeddings for large clusters
    template <typename T>
    static std::vector<std::vector<T>> calculateClusterMeans(const std::vector<std::vector<T>>& embeddings,
                                                 const std::vector<int>& clusters,
                                                 const std::vector<int>& largeClusters) 
    {
        std::vector<std::vector<T>> clusterMeans;

        for (int large_k : largeClusters) {
            std::vector<T> meanEmbedding( embeddings[0].size(), 0.0 );
            int count = 0;
            for (size_t i = 0; i < clusters.size(); ++i) {
                if (clusters[i] == large_k) {
                    // Add the embedding to the mean
                    for (size_t j = 0; j < meanEmbedding.size(); ++j) {
                        meanEmbedding[j] += embeddings[i][j];
                    }
                    count++;
                }
            }

            // Calculate the mean by dividing by the count
            if (count > 0) {
                for (size_t j = 0; j < meanEmbedding.size(); ++j) {
                    meanEmbedding[j] /= static_cast<T>(count);
                }
            }

            clusterMeans.push_back(meanEmbedding);
        }

        return clusterMeans;
    }

    // Function to calculate the cosine distance between two vectors
    template <typename T>
    static T cosineDistance(const std::vector<T>& vec1, const std::vector<T>& vec2) 
    {
        if (vec1.size() != vec2.size()) {
            throw std::runtime_error("Vector sizes must be equal.");
        }

        T dotProduct = 0.0;
        T magnitude1 = 0.0;
        T magnitude2 = 0.0;

        for (size_t i = 0; i < vec1.size(); ++i) {
            dotProduct += static_cast<T>(vec1[i]) * static_cast<T>(vec2[i]);
            magnitude1 += static_cast<T>(vec1[i]) * static_cast<T>(vec1[i]);
            magnitude2 += static_cast<T>(vec2[i]) * static_cast<T>(vec2[i]);
        }

        if (magnitude1 == 0.0 || magnitude2 == 0.0) {
            throw std::runtime_error("Vectors have zero magnitude.");
        }

        return 1.0 - (dotProduct / (std::sqrt(magnitude1) * std::sqrt(magnitude2)));
    }

    // Calculate cosine distances between large and small cluster means
    template <typename T>
    static std::vector<std::vector<T>> cosineSimilarity( std::vector<std::vector<T>>& largeClusterMeans,
            std::vector<std::vector<T>>& smallClusterMeans )
    {

        std::vector<std::vector<T>> centroidsCdist( largeClusterMeans.size(),
                std::vector<T>( smallClusterMeans.size()));
        for (size_t i = 0; i < largeClusterMeans.size(); ++i) {
            for (size_t j = 0; j < smallClusterMeans.size(); ++j) {
                T distance = cosineDistance(largeClusterMeans[i], smallClusterMeans[j]);
                centroidsCdist[i][j] = distance;
            }
        }

        return centroidsCdist;
    }

    // Function to find unique clusters and return the inverse mapping
    static std::vector<int> findUniqueClusters(const std::vector<int>& clusters,
                                        std::vector<int>& uniqueClusters) 
    {
        std::vector<int> inverseMapping(clusters.size(), -1);
        int nextClusterIndex = 0;

        // Find unique
        for (size_t i = 0; i < clusters.size(); ++i) 
        {
            std::vector<int>::iterator position = std::find( uniqueClusters.begin(), uniqueClusters.end(), clusters[i] );
            if (position == uniqueClusters.end()) 
            {
                uniqueClusters.push_back(clusters[i]);
            }
        }

        // Sort, python implementation like this
        std::sort(uniqueClusters.begin(), uniqueClusters.end());

        for (size_t i = 0; i < clusters.size(); ++i) 
        {
            std::vector<int>::iterator position = std::find( uniqueClusters.begin(), uniqueClusters.end(), clusters[i] );
            if (position != uniqueClusters.end()) 
            {
                inverseMapping[i] = position - uniqueClusters.begin();
            }
        }

        return inverseMapping;
    }

    // python: embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)
    template <typename T>
    static std::vector<std::vector<std::vector<T>>> rearrange_up( const std::vector<std::vector<T>>& input, int c )
    {
        assert( input.size() > c );
        assert( input.size() % c == 0 );
        size_t dim1 = c;
        size_t dim2 = input.size() / c;
        size_t dim3 = input[0].size();
        std::vector<std::vector<std::vector<T>>> output(c, 
                std::vector<std::vector<T>>( dim2, std::vector<T>(dim3, -1.0f)));
        for( size_t i = 0; i < dim1; ++i )
        {
            for( size_t j = 0; j < dim2; ++j )
            {
                for( size_t k = 0; k < dim3; ++k )
                {
                    output[i][j][k] = input[i*dim2+j][k];
                }
            }
        }

        return output;
    }

    // Imlemenation of einops.rearrange c s d -> (c s) d
    template <typename T>
    static std::vector<std::vector<T>> rearrange_down( const std::vector<std::vector<std::vector<T>>>& input )
    {
        int num_chunks = input.size();
        int num_frames = input[0].size();
        int num_classes = input[0][0].size();
        std::vector<std::vector<T>> data(num_chunks * num_frames, std::vector<T>(num_classes));
        for( size_t i = 0; i < num_chunks; ++i )
        {
            for( size_t j = 0; j < num_frames; ++j )
            {
                for( size_t k = 0; k < num_classes; ++k )
                {
                    data[ i * num_frames + j][k] = input[i][j][k];
                }
            }
        }

        return data;
    }

    // Imlemenation of einops.rearrange c f k -> (c k) f
    template <typename T>
    static std::vector<std::vector<T>> rearrange_other( const std::vector<std::vector<std::vector<T>>>& input )
    {
        int num_chunks = input.size();
        int num_frames = input[0].size();
        int num_classes = input[0][0].size();
        std::vector<std::vector<T>> data(num_chunks * num_classes, std::vector<T>(num_frames));
        int rowNum = 0;
        for ( const auto& row : input ) 
        {
            // Create a new matrix with swapped dimensions
            std::vector<std::vector<T>> transposed(num_classes, std::vector<T>(num_frames));

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < num_classes; ++j) {
                    data[rowNum * num_classes + j][i] = row[i][j];
                }
            }

            rowNum++;
        }

        return data;
    }

    static std::vector<std::vector<int>> wellDefinedIndex( const std::vector<std::vector<bool>>& off_or_on ) 
    {
        // Find the indices of True values in each row and store them in a vector of vectors
        size_t max_indices = 0;
        std::vector<std::vector<int>> nonzero_indices;
        for (const auto& row : off_or_on) {
            std::vector<int> indices = nonzeroIndices(row);
            if( indices.size() > max_indices )
                max_indices = indices.size();
            nonzero_indices.push_back(indices);
        }

        // Fill missing indices with -1 and create the well_defined_idx vector of vectors
        std::vector<std::vector<int>> well_defined_idx;
        for (const auto& indices : nonzero_indices) {
            if( indices.size() < max_indices )
            {
                std::vector<int> filled_indices(max_indices, -1);
                std::copy(indices.begin(), indices.end(), filled_indices.begin());
                well_defined_idx.push_back(filled_indices);
            }
            else
            {
                well_defined_idx.push_back( indices );
            }
        }

        return well_defined_idx;
    }

    // Function to calculate cumulative sum along axis=1
    static std::vector<std::vector<int>> cumulativeSum(const std::vector<std::vector<bool>>& input) 
    {
        std::vector<std::vector<int>> cumsum;

        for (const auto& row : input) {
            std::vector<int> row_cumsum;
            int running_sum = 0;

            for (bool val : row) {
                running_sum += val ? 1 : 0;
                row_cumsum.push_back(running_sum);
            }

            cumsum.push_back(row_cumsum);
        }

        return cumsum;
    }

    // Define a helper function to calculate np.where
    static std::vector<std::vector<bool>> numpy_where(const std::vector<std::vector<int>>& same_as,
                                    const std::vector<std::vector<bool>>& on,
                                    const std::vector<std::vector<int>>& well_defined_idx,
                                    const std::vector<std::vector<bool>>& initial_state,
                                    const std::vector<std::vector<int>>& samples) 
    {
        assert( same_as.size() == on.size());
        assert( same_as.size() == well_defined_idx.size());
        assert( same_as.size() == initial_state.size());
        assert( same_as.size() == samples.size());
        assert( same_as[0].size() == on[0].size());
        assert( same_as[0].size() == well_defined_idx[0].size());
        assert( same_as[0].size() == initial_state[0].size());
        assert( same_as[0].size() == samples[0].size());
        std::vector<std::vector<bool>> result( same_as.size(), std::vector<bool>( same_as[0].size(), false ));
        for( size_t i = 0; i < same_as.size(); ++i )
        {
            for( size_t j = 0; j < same_as[0].size(); ++j )
            {
                if( same_as[i][j] > 0 )
                {
                    int x = samples[i][j];
                    int y = well_defined_idx[x][same_as[i][j]-1];
                    result[i][j] = on[x][y];
                }
                else
                {
                    result[i][j] = initial_state[i][j];
                }
            }
        }


        return result;
    }

    static std::vector<std::vector<std::vector<double>>> cleanSegmentations(
            const std::vector<std::vector<std::vector<double>>>& data)
    {
        size_t numRows = data.size();
        size_t numCols = data[0].size();
        size_t numChannels = data[0][0].size();

        // Initialize the result with all zeros
        std::vector<std::vector<std::vector<double>>> result(numRows, 
                std::vector<std::vector<double>>(numCols, std::vector<double>(numChannels, 0.0)));
        for (int i = 0; i < numRows; ++i) 
        {
            for (int j = 0; j < numCols; ++j) 
            {
                double sum = 0.0;
                for (int k = 0; k < numChannels; ++k) 
                {
                    sum += data[i][j][k];
                }
                bool keep = false;
                if( sum < 2.0 )
                {
                    keep = true;
                }
                for (int k = 0; k < numChannels; ++k) 
                {
                    if( keep )
                        result[i][j][k] = data[i][j][k];
                }
            }
        }

        return result;
    }

    // Define a function to interpolate 2D arrays (nearest-neighbor interpolation)
    static std::vector<std::vector<bool>> interpolate(const std::vector<std::vector<float>>& masks, 
            int num_samples, float threshold ) 
    {
        int inputHeight = masks.size();
        int inputWidth = masks[0].size();

        std::vector<std::vector<bool>> output(inputHeight, std::vector<bool>(num_samples, false));
        assert( num_samples > inputWidth );
        int scale = num_samples / inputWidth;

        for (int i = 0; i < inputHeight; ++i) 
        {
            for (int j = 0; j < num_samples; ++j) 
            {
                int src_y = j * inputWidth / num_samples;
                if( masks[i][src_y] > threshold )
                    output[i][j] = true;
            }
        }

        return output;
    }

    // Define a function to perform pad_sequence
    static std::vector<std::vector<float>> padSequence(const std::vector<std::vector<float>>& waveforms,
                                                const std::vector<std::vector<bool>>& imasks) 
    {
        // Find the maximum sequence length
        size_t maxLen = 0;
        for (const std::vector<bool>& mask : imasks) 
        {
            maxLen = std::max(maxLen, mask.size());
        }

        // Initialize the padded sequence with zeros
        std::vector<std::vector<float>> paddedSequence(waveforms.size(), std::vector<float>(maxLen, 0.0));

        // Copy the valid data from waveforms based on imasks
        for (size_t i = 0; i < waveforms.size(); ++i) 
        {
            size_t validIndex = 0;
            for (size_t j = 0; j < imasks[i].size(); ++j) 
            {
                if (imasks[i][j]) 
                {
                    paddedSequence[i][validIndex++] = waveforms[i][j];
                }
            }
        }

        return paddedSequence;
    }

};


class Segment
{
public:
    double start;
    double end;
    Segment( double start, double end )
        : start( start )
        , end( end )
    {}

    Segment( const Segment& other )
    {
        start = other.start;
        end = other.end;
    }

    Segment& operator=( const Segment& other )
    {
        start = other.start;
        end = other.end;

        return *this;
    }

    double duration() const 
    {
        return end - start;
    }

    double gap( const Segment& other )
    {
        if( start < other.start )
        {
            if( end >= other.start )
            {
                return 0.0;
            }
            else
            {
                return other.start - end;
            }
        }
        else
        {
            if( start <= other.end )
            {
                return 0.0;
            }
            else
            {
                return start - other.end;
            }
        }
    }

    Segment merge( const Segment& other )
    {
        return Segment(std::min( start, other.start ), std::max( end, other.end ));
    }
};

// Define a struct to represent annotations
struct Annotation 
{
    struct Result
    {
        double start;
        double end;
        int label;
        Result( double start, double end, int label )
            : start( start )
            , end( end )
            , label( label )
        {}
    };
    struct Track 
    {
        std::vector<Segment> segments;
        int label;

        Track( int label )
            : label( label )
        {}

        Track& operator=( const Track& other )
        {
            segments = other.segments;
            label = other.label;

            return *this;
        }

        Track( const Track& other )
        {
            segments = other.segments;
            label = other.label;
        }

        Track( Track&& other )
        {
            segments = std::move( other.segments );
            label = other.label;
        }

        void addSegment(double start, double end ) 
        {
            segments.push_back( Segment( start, end ));
        }

        void support( double collar )
        {
            // Must sort first
            std::sort( segments.begin(), segments.end(), []( const Segment& s1, const Segment& s2 ){
                        return s1.start < s2.start;
                    });
            if( segments.size() == 0 )
                return;
            std::vector<Segment> merged_segments;
            Segment curSeg = segments[0];
            bool merged = true;
            for( size_t i = 1; i < segments.size(); ++i )
            {
                // WHYWHY must assign to tmp object, otherwise
                // in gap function, its value like random
                auto next = segments[i];
                double gap = curSeg.gap( next );
                if( gap < collar )
                {
                    curSeg = curSeg.merge( segments[i] );
                }
                else
                {
                    merged_segments.push_back( curSeg );
                    curSeg = segments[i];
                }
            }
            merged_segments.push_back( curSeg );

            segments.swap( merged_segments );
        }

        void removeShort( double min_duration_on )
        {
            for( size_t i = 1; i < segments.size(); ++i )
            {
                if( segments[i].duration() < min_duration_on )
                {
                    segments.erase( segments.begin() + i );
                    i--;
                }
            }
        }
    };

    std::vector<Track> tracks;

    Annotation()
        : tracks()
    {}

    std::vector<Result> finalResult()
    {
        std::vector<Result> results;
        for( const auto& track : tracks )
        {
            for( const auto& segment : track.segments )
            {
                Result res( segment.start, segment.end, track.label );
                results.push_back( res );
            }
        }
        std::sort( results.begin(), results.end(), []( const Result& s1, const Result& s2 ){
                    return s1.start < s2.start;
                });

        return results;
    }

    void addSegment(double start, double end, int label) 
    {
        for( auto& tk : tracks )
        {
            if( tk.label == label )
            {
                tk.addSegment( start, end );
                return;
            }
        }

        // Not found, create new track
        Track tk( label );
        tk.addSegment( start, end );
        tracks.push_back( tk );
    }

    Annotation& operator=( const Annotation& other )
    {
        tracks = other.tracks;

        return *this;
    }

    Annotation( Annotation&& other )
    {
        tracks = std::move( tracks );
    }

    void removeShort( double min_duration_on )
    {
        for( auto& track : tracks )
        {
            track.removeShort( min_duration_on );
        }
    }

    // pyannote/core/annotation.py:1350
    void support( double collar )
    {
        // python: timeline = timeline.support(collar)
        // pyannote/core/timeline.py:845
        for( auto& track : tracks )
        {
            track.support( collar );
        }
    }
};

class SlidingWindow
{
public:
    double start;
    double step;
    double duration;
    size_t num_samples;
    double sample_rate;
    SlidingWindow()
        : start( 0.0 )
        , step( 0.0 )
        , duration( 0.0 )
        , num_samples( 0 )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( size_t num_samples )
        : start( 0.0 )
        , step( 0.0 )
        , duration( 0.0 )
        , num_samples( num_samples )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( double start, double step, double duration, size_t num_samples = 0 )
        : start( start )
        , step( step )
        , duration( duration )
        , num_samples( num_samples )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( const SlidingWindow& other )
    {
        start = other.start;
        step = other.step;
        duration = other.duration;
        num_samples = other.num_samples;
        sample_rate = other.sample_rate;
    }

    SlidingWindow& operator=( const SlidingWindow& other )
    {
        start = other.start;
        step = other.step;
        duration = other.duration;
        num_samples = other.num_samples;
        sample_rate = other.sample_rate;

        return *this;
    }

    size_t closest_frame( double start )
    {
        double closest = ( start - this->start - .5 * duration ) / step;
        if( closest < 0.0 )
            closest = 0.0;
        return Helper::np_rint( closest );
    }

    Segment operator[]( int pos ) const
    {
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        // python: start = self.__start + i * self.__step
        //double start = this->start + pos * this->step;
        double start = 0.0;
        size_t cur_frames = 0;
        int index = 0;
        while( true )
        {
            if( index == pos )
                return Segment( start, start + duration );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += step;
            cur_frames += step_size;
            index++;
        }

        return Segment(0.0, 0.0);
    }

    std::vector<Segment> data()
    {
        std::vector<Segment> segments;
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        double start = 0.0;
        size_t cur_frames = 0;
        while( true )
        {
            Segment seg( start, start + duration );
            segments.push_back( seg );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += step;
            cur_frames += step_size;
        }

        return segments;
    }
};

class SlidingWindowFeature
{
public:
    std::vector<std::vector<std::vector<float>>> data;
    std::vector<std::pair<double, double>> slidingWindow;

    SlidingWindowFeature& operator=( const SlidingWindowFeature& other )
    {
        data = other.data;
        slidingWindow = other.slidingWindow;

        return *this;
    }

    SlidingWindowFeature( const SlidingWindowFeature& other )
    {
        data = other.data;
        slidingWindow = other.slidingWindow;
    }
};

class PipelineHelper
{
public:
    // pyannote/audio/core/inference.py:411
    // we ignored warm_up parameter since our case use default value( 0.0, 0.0 )
    // so hard code warm_up
    static std::vector<std::vector<double>> aggregate( 
            const std::vector<std::vector<std::vector<double>>>& scoreData, 
            const SlidingWindow& scores_frames, 
            const SlidingWindow& pre_frames, 
            SlidingWindow& post_frames,
            bool hamming = false, 
            double missing = NAN, 
            bool skip_average = false,
            double epsilon = std::numeric_limits<double>::epsilon())
    {
        size_t num_chunks = scoreData.size(); 
        size_t num_frames_per_chunk = scoreData[0].size(); 
        size_t num_classes = scoreData[0][0].size(); 
        size_t num_samples = scores_frames.num_samples;
        assert( num_samples > 0 );

        // create masks 
        std::vector<std::vector<std::vector<double>>> masks( num_chunks, 
                std::vector<std::vector<double>>( num_frames_per_chunk, std::vector<double>( num_classes, 1.0 )));
        auto scores = scoreData;

        // Replace NaN values in scores with 0 and update masks
        // python: masks = 1 - np.isnan(scores)
        // python: scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)
        for (size_t i = 0; i < num_chunks; ++i) 
        {
            for (size_t j = 0; j < num_frames_per_chunk; ++j) 
            {
                for( size_t k = 0; k < num_classes; ++k )
                {
                    if (std::isnan(scoreData[i][j][k])) 
                    {
                        masks[i][j][k] = 0.0;
                        scores[i][j][k] = 0.0;
                    }
                }
            }
        }

        if( !hamming )
        {
            // python np.ones((num_frames_per_chunk, 1))
            // no need create it, later will directly apply 1 to computation
        }
        else
        {
            // python: np.hamming(num_frames_per_chunk).reshape(-1, 1)
            assert( false ); // no implemented
        }

        // Get frames, we changed this part. In pyannote, it calc frames(self._frames) before calling
        // this function, but in this function, it creates new frames and use it.
        // step = (self.inc_num_samples / self.inc_num_frames) / sample_rate
        // pyannote/audio/core/model.py:243
        // currently cannot find where self.inc_num_samples / self.inc_num_frames from
        /*int inc_num_samples = 270; // <-- this may not be correct
        int inc_num_frames = 1; // <-- this may not be correct
        float frames_step = ( inc_num_samples * 1.0f / inc_num_frames) / g_sample_rate;
        float frames_duration = frames_step;
        float frames_start = scores_frames.start;
        */

        // aggregated_output[i] will be used to store the sum of all predictions
        // for frame #i
        // python: num_frames = ( frames.closest_frame(...)) + 1
        double frame_target = scores_frames.start + scores_frames.duration + (num_chunks - 1) * scores_frames.step;
        SlidingWindow frames( scores_frames.start, pre_frames.step, pre_frames.duration );
        size_t num_frames = frames.closest_frame( frame_target ) + 1;

        // python: aggregated_output: np.ndarray = np.zeros(...
        std::vector<std::vector<double>> aggregated_output(num_frames, std::vector<double>( num_classes, 0.0 ));

        // overlapping_chunk_count[i] will be used to store the number of chunks
        // that contributed to frame #i
        std::vector<std::vector<double>> overlapping_chunk_count(num_frames, std::vector<double>( num_classes, 0.0 ));

        // aggregated_mask[i] will be used to indicate whether
        // at least one non-NAN frame contributed to frame #i
        std::vector<std::vector<double>> aggregated_mask(num_frames, std::vector<double>( num_classes, 0.0 ));
        
        // for our use case, warm_up_window and hamming_window all 1
        double start = scores_frames.start;
        for( size_t i = 0; i < scores.size(); ++i )
        {
            size_t start_frame = frames.closest_frame( start );
            std::cout<<"start_frame: "<<start_frame<<" with:"<<start<<std::endl;
            start += scores_frames.step; // python: chunk.start
            for( size_t j = 0; j < num_frames_per_chunk; ++j )
            {
                size_t _j = j + start_frame;
                for( size_t k = 0; k < num_classes; ++k )
                {
                    // score * mask * hamming_window * warm_up_window
                    aggregated_output[_j][k] += scores[i][j][k] * masks[i][j][k];
                    overlapping_chunk_count[_j][k] += masks[i][j][k];
                    if( masks[i][j][k] > aggregated_mask[_j][k] )
                    {
                        aggregated_mask[_j][k] = masks[i][j][k];
                    }
                }
            }
        }

#ifdef WRITE_DATA
        debugWrite3d( masks, "cpp_masks_in_aggregate" );
        debugWrite3d( scores, "cpp_scores_in_aggregate" );
        debugWrite2d( aggregated_output, "cpp_aggregated_output" );
        debugWrite2d( aggregated_mask, "cpp_aggregated_mask" );
        debugWrite2d( overlapping_chunk_count, "cpp_overlapping_chunk_count" );
#endif // WRITE_DATA

        post_frames.start = frames.start;
        post_frames.step = frames.step;
        post_frames.duration = frames.duration;
        post_frames.num_samples = num_samples;
        if( !skip_average )
        {
            for( size_t i = 0; i < aggregated_output.size(); ++i )
            {
                for( size_t j = 0; j < aggregated_output[0].size(); ++j )
                {
                    aggregated_output[i][j] /= std::max( overlapping_chunk_count[i][j], epsilon );
                }
            }
        }
        else
        {
            // do nothing
        }

        // average[aggregated_mask == 0.0] = missing
        for( size_t i = 0; i < aggregated_output.size(); ++i )
        {
            for( size_t j = 0; j < aggregated_output[0].size(); ++j )
            {
                if( abs( aggregated_mask[i][j] ) < std::numeric_limits<double>::epsilon() )
                {
                    aggregated_output[i][j] = missing;
                }
            }
        }

        return aggregated_output;

    }

};

struct DisSeg
{
    int i;
    int j;
    bool operator==( const DisSeg& other )
    {
        if( i == other.i && j == other.j )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

class SegmentModel : public OnnxModel 
{
private:
    double m_duration = 5.0;
    double m_step = 0.5;
    int m_batch_size = 32;
    int m_sample_rate = 16000;
    double m_diarization_segmentation_threashold = 0.4442333667381752;
    double m_diarization_segmentation_min_duration_off = 0.5817029604921046;
    size_t m_num_samples = 0;


public:
    SegmentModel(const std::string& model_path)
        : OnnxModel(model_path) {
    }


    // input: batch size x channel x samples count, for example, 32 x 1 x 80000
    // output: batch size x 293 x 3
    std::vector<std::vector<std::vector<float>>> infer( const std::vector<std::vector<float>>& waveform )
    {
        // Create a std::vector<float> with the same size as the tensor
        //std::vector<float> audio( waveform.size() * waveform[0].size());
        std::vector<float> audio( m_batch_size * waveform[0].size());
        //for( size_t i = 0; i < waveform.size(); ++i )
        for( size_t i = 0; i < m_batch_size; ++i )
        {
            for( size_t j = 0; j < waveform[0].size(); ++j )
            {
                audio[i*waveform[0].size() + j] = waveform[i][j];
            }
        }

        // batch_size * num_channels (1 for mono) * num_samples
        //const int64_t batch_size = waveform.size();
        const int64_t batch_size = m_batch_size;
        const int64_t num_channels = 1;
        int64_t input_node_dims[3] = {batch_size, num_channels,
            static_cast<int64_t>(waveform[0].size())};
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(audio.data()), audio.size(),
                input_node_dims, 3);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_ort));

        auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
                ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

        const float* outputs = ort_outputs[0].GetTensorData<float>();
        auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int len1 = outputs_shape[0];
        int len2 = outputs_shape[1];
        int len3 = outputs_shape[2];
        std::cout<<"output shape:"<<len1<<"x"<<len2<<"x"<<len3<<std::endl;

        len1 = waveform.size();  // <====
        std::vector<std::vector<std::vector<float>>> res( len1, 
                std::vector<std::vector<float>>( len2, std::vector<float>( len3 )));
        for( int i = 0; i < len1; ++i )
        {
            for( int j = 0; j < len2; ++j )
            {
                for( int k = 0; k < len3; ++k )
                {
                    res[i][j][k] = *( outputs + i * len2 * len3 + j * len3 + k );
                }
            }
        }

        return res;
    }

    // pyannote/audio/core/inference.py:202
    std::vector<std::vector<std::vector<float>>> slide(const std::vector<float>& waveform, 
            SlidingWindow& res_frames )
    {
        int sample_rate = 16000;
        int window_size = std::round(m_duration * sample_rate); // 80000
        int step_size = std::round(m_step * sample_rate); // 8000
        int num_channels = 1;
        size_t num_samples = waveform.size();
        int num_frames_per_chunk = 293; // Need to check with multiple wave files
        size_t i = 0;
        std::vector<std::vector<float>> chunks;
        std::vector<std::vector<std::vector<float>>> outputs;
        while( i + window_size < num_samples )
        {
            // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = start + window_size;

            // To store the sliced vector
            std::vector<float> chunk( window_size, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            if( chunks.size() == m_batch_size )
            {
                auto tmp = infer( chunks );
                for( const auto& a : tmp )
                {
                    outputs.push_back( a );
                }
                chunks.clear();
            }

            i += step_size;
        }

        // Process remaining chunks
        if( chunks.size() > 0 )
        {
            auto tmp = infer( chunks );
            for( const auto& a : tmp )
            {
                outputs.push_back( a );
            }
            chunks.clear();
        }

        // Process last chunk if have, last chunk may not equal window_size
        // Make sure at least we have 1 element remaining
        if( i + 1 < num_samples )
        {
            // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = waveform.end();

            // To store the sliced vector, always window_size, for last chunk we pad with 0.0
            std::vector<float> chunk( end - start, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            auto tmp = infer( chunks );
            assert( tmp.size() == 1 );

            // Padding
            auto a = tmp[0];
            for( size_t i = a.size(); i < num_frames_per_chunk;  ++i )
            {
                std::vector<float> pad( a[0].size(), 0.0 );
                a.push_back( pad );
            }
            outputs.push_back( a );
        }

        // Calc segments
        res_frames.start = 0.0;
        res_frames.step = m_step;
        res_frames.duration = m_duration;
        res_frames.num_samples = num_samples;
        /*
        float start = 0.0;
        size_t cur_frames = 0;
        while( true )
        {
            std::pair<float, float> seg = { start, start + m_duration };
            segments.push_back( seg );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += m_step;
            cur_frames += step_size;
        }
        */

        return outputs;
    }

    std::vector<std::vector<std::vector<double>>> binarize_swf(
        const std::vector<std::vector<std::vector<float>>> scores,
        bool initial_state = false ) 
    {
        double onset = m_diarization_segmentation_threashold;

        // TODO: use hlper::rerange_down
        // Imlemenation of einops.rearrange c f k -> (c k) f
        int num_chunks = scores.size();
        int num_frames = scores[0].size();
        int num_classes = scores[0][0].size();
        std::vector<std::vector<double>> data(num_chunks * num_classes, std::vector<double>(num_frames));
        int rowNum = 0;
        for ( const auto& row : scores ) 
        {
            // Create a new matrix with swapped dimensions
            std::vector<std::vector<double>> transposed(num_classes, std::vector<double>(num_frames));

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < num_classes; ++j) {
                    data[rowNum * num_classes + j][i] = row[i][j];
                }
            }

            rowNum++;
        }
        /*
        for( const auto& d : data )
        {
            for( float e : d )
            {
                std::cout<<e<<",";
            }
            std::cout<<std::endl;
        }
        */

        auto binarized = binarize_ndarray( data, onset, initial_state);

        // TODO: use help::rerange_up
        // Imlemenation of einops.rearrange (c k) f -> c f k - restore
        std::vector<std::vector<std::vector<double>>> restored(num_chunks, 
                std::vector<std::vector<double>>( num_frames, std::vector<double>(num_classes)));
        rowNum = 0;
        for( size_t i = 0; i < binarized.size(); i += num_classes )
        {
            for( size_t j = 0; j < num_classes; ++j )
            {
                for( size_t k = 0; k < num_frames; ++k )
                {
                    restored[rowNum][k][j] = binarized[i+j][k];
                }
            }
            rowNum++;
        }

        return restored;
    }

    std::vector<std::vector<bool>> binarize_ndarray(
        const std::vector<std::vector<double>>& scores,
        double onset = 0.5,
        bool initialState = false
    ) {

        // Scores shape like 2808x293
        size_t rows = scores.size();
        size_t cols = scores[0].size();

        // python: on = scores > onset
        // on is same shape as scores, with true or false inside
        std::vector<std::vector<bool>> on( rows, std::vector<bool>( cols, false ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                if( scores[i][j] > onset )
                    on[i][j] = true;
            }
        }

        // python: off_or_on = (scores < offset) | on
        // off_or_on is same shape as scores, with true or false inside
        // Since onset and offset is same value, it should be true unless score[i][j] == onset
        std::vector<std::vector<bool>> off_or_on( rows, std::vector<bool>( cols, true ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                if(abs( scores[i][j] - onset ) < std::numeric_limits<double>::epsilon())
                    off_or_on[i][j] = false;
            }
        }

        // python: # indices of frames for which the on/off state is well-defined
        // well_defined_idx = np.array(
        //     list(zip_longest(*[np.nonzero(oon)[0] for oon in off_or_on], fillvalue=-1))
        // ).T
        auto well_defined_idx = Helper::wellDefinedIndex( off_or_on );

        // same_as same shape of as scores
        // python: same_as = np.cumsum(off_or_on, axis=1)
        auto same_as = Helper::cumulativeSum( off_or_on );

        // python: samples = np.tile(np.arange(batch_size), (num_frames, 1)).T
        std::vector<std::vector<int>> samples( rows, std::vector<int>( cols, 0 ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                samples[i][j] = i;
            }
        }

        // create same shape of initial_state as scores.
        std::vector<std::vector<bool>> initial_state( rows, std::vector<bool>( cols, initialState ));


        // python: return np.where( same_as, on[samples, well_defined_idx[samples, same_as - 1]], initial_state)
        // TODO: delete tmp, directly return
#ifdef WRITE_DATA
        debugWrite2d( scores, "cpp_binarize_score" );
        debugWrite2d( same_as, "cpp_same_as" );
        debugWrite2d( on, "cpp_on" );
        debugWrite2d( well_defined_idx, "cpp_well_defined_idx" );
        debugWrite2d( initial_state, "cpp_initial_state" );
        debugWrite2d( samples, "cpp_samples" );
#endif // WRITE_DATA
        auto tmp = Helper::numpy_where( same_as, on, well_defined_idx, initial_state, samples );
#ifdef WRITE_DATA
        debugWrite2d( tmp, "cpp_binary_ndarray" );
#endif // WRITE_DATA
        return tmp;
    }

    std::vector<float> crop( const std::vector<float>& waveform, std::pair<double, double> segment) 
    {
        int start_frame = static_cast<int>(std::floor(segment.first * m_sample_rate));
        int frames = static_cast<int>(waveform.size());

        int num_frames = static_cast<int>(std::floor(m_duration * m_sample_rate));
        int end_frame = start_frame + num_frames;

        int pad_start = -std::min(0, start_frame);
        int pad_end = std::max(end_frame, frames) - frames;
        start_frame = std::max(0, start_frame);
        end_frame = std::min(end_frame, frames);
        num_frames = end_frame - start_frame;

        std::vector<float> data(waveform.begin() + start_frame, waveform.begin() + end_frame);

        // Pad with zeros
        data.insert(data.begin(), pad_start, 0.0);
        data.insert(data.end(), pad_end, 0.0);

        return data;
    }

    // pyannote/audio/pipelines/utils/diarization.py:108
    std::vector<int> speaker_count( const std::vector<std::vector<std::vector<float>>>& segmentations,
            const std::vector<std::vector<std::vector<double>>>& binarized,
            const SlidingWindow& pre_frame,
            SlidingWindow& count_frames,
            int num_samples )
    {
        // Get frames first - python: self._frames
        // step = (self.inc_num_samples / self.inc_num_frames) / sample_rate
        // pyannote/audio/core/model.py:243
        // currently cannot where where self.inc_num_samples / self.inc_num_frames from
        //int inc_num_samples = 270; // <-- this may not be correct
        //int inc_num_frames = 1; // <-- this may not be correct
        //float step = ( inc_num_samples * 1.0f / inc_num_frames) / m_sample_rate;
        //float window = step;
        //std::vector<std::pair<float, float>> frames;
        //float start = 0.0;
        //while( true )
        //{
        //    start += step;
        //    if( start * m_sample_rate >= num_samples )
        //        break;
        //    float end = start + window;
        //    frames.emplace_back(std::make_pair<float, float>(start, end));
        //}

        // python: trimmed = Inference.trim
        SlidingWindow trimmed_frames;
        SlidingWindow frames( 0.0, m_step, m_duration );
        auto trimmed = trim( binarized, 0.1, 0.1, frames, trimmed_frames );

#ifdef WRITE_DATA
        debugWrite3d( trimmed, "cpp_trimmed" );
#endif // WRITE_DATA

        // python: count = Inference.aggregate(
        // python: np.sum(trimmed, axis=-1, keepdims=True)
        std::vector<std::vector<std::vector<double>>> sum_trimmed( trimmed.size(), 
                std::vector<std::vector<double>>( trimmed[0].size(), std::vector<double>( 1 )));
        for( size_t i = 0; i < trimmed.size(); ++i )
        {
            for( size_t j = 0; j < trimmed[0].size(); ++j )
            {
                double sum = 0.0;
                for( size_t k = 0; k < trimmed[0][0].size(); ++k )
                {
                    sum += trimmed[i][j][k];
                }
                sum_trimmed[i][j][0] = sum;
            }
        }
#ifdef WRITE_DATA
        debugWrite3d( sum_trimmed, "cpp_sum_trimmed" );
#endif // WRITE_DATA
       
        auto count_data = PipelineHelper::aggregate( sum_trimmed, trimmed_frames, 
                pre_frame, count_frames, false, 0.0, false );

#ifdef WRITE_DATA
        debugWrite2d( count_data, "cpp_count_data" );
#endif // WRITE_DATA
       
        // count_data is Nx1, so we convert it to 1d array
        assert( count_data[0].size() == 1 );

        // python: count.data = np.rint(count.data).astype(np.uint8)
        //std::vector<std::vector<int>> res( count_data.size(), std::vector<int>( count_data[0].size()));
        std::vector<int> res( count_data.size());
        for( size_t i = 0; i < res.size(); ++i )
        {
            res[i] = Helper::np_rint( count_data[i][0] );
        }

        return res;
    }

    // pyannote/audio/core/inference.py:540
    // use after_trim_step, after_trim_duration to calc sliding_window later 
    std::vector<std::vector<std::vector<double>>> trim(
            const std::vector<std::vector<std::vector<double>>>& binarized, 
            double left, double right, 
            const SlidingWindow& before_trim, 
            SlidingWindow& trimmed_frames )
    {
        double before_trim_start = before_trim.start;
        double before_trim_step = before_trim.step;
        double before_trim_duration = before_trim.duration;
        size_t chunkSize = binarized.size();
        size_t num_frames = binarized[0].size();

        // python: num_frames_left = round(num_frames * warm_up[0])
        size_t num_frames_left = floor(num_frames * left);

        // python: num_frames_right = round(num_frames * warm_up[1])
        size_t num_frames_right = floor(num_frames * right);
        size_t num_frames_step = floor(num_frames * before_trim_step / before_trim_duration);

        // python: new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]
        std::vector<std::vector<std::vector<double>>> trimed( binarized.size(), 
                std::vector<std::vector<double>>((num_frames - num_frames_right - num_frames_left), 
                std::vector<double>( binarized[0][0].size())));
        for( size_t i = 0; i < binarized.size(); ++i )
        {
            for( size_t j = num_frames_left; j < num_frames - num_frames_right; ++j )
            {
                for( size_t k = 0; k < binarized[0][0].size(); ++k )
                {
                    trimed[i][j - num_frames_left][k] = binarized[i][j][k];
                }
            }
        }

        trimmed_frames.start = before_trim_start + left * before_trim_duration;
        trimmed_frames.step = before_trim_step;
        trimmed_frames.duration = ( 1 - left - right ) * before_trim_duration;
        trimmed_frames.num_samples = num_frames - num_frames_right - num_frames_left;

        return trimed;
    }

}; // SegmentModel


class EmbeddingModel : public OnnxModel 
{
private:
    size_t m_batchSize = 32;


public:
    EmbeddingModel(const std::string& model_path)
        : OnnxModel(model_path) {
    }

    /*
     * input: batch size x samples count, and wav_lens is 1d array. for example, 32 x 80000,
     * output: batch size x 192 
     * Note, the model we exported only can generate correct result when batch size is 32.
     * so here we padding data to 32. Ideally, it should generate expected result with 
     * any batch size because we set batch size as dynamic axs when export, but it does 
     * not work, it seems something wrong with MyEmbedding class model in embedding/myembedding.py
     *
     * If we can fix above issue, then no need pad data here. Another solution is to 
     * when export model use batch size 1, then iterate batch data one by one, but it will 
     * spend more time for processing whole batch.
     * */
    std::vector<std::vector<float>> infer( const std::vector<std::vector<float>>& data,
            const std::vector<float>& lens )
    {
        assert( data.size() == lens.size());
        assert( data.size() <= m_batchSize );
        std::vector<std::vector<float>> waveform( m_batchSize, std::vector<float>( data[0].size(), 0.0 ));
        std::vector<float> wav_lens( m_batchSize, 1.0 );

        // TODO: no need copy, directly copy from data
        std::copy( data.begin(), data.end(), waveform.begin());
        std::copy( lens.begin(), lens.end(), wav_lens.begin());

        // Create a std::vector<float> with the same size as the tensor
        std::vector<float> audio( waveform.size() * waveform[0].size());
        size_t row_len = waveform[0].size();
        for( size_t i = 0; i < waveform.size(); ++i )
        {
            for( size_t j = 0; j < row_len; ++j )
            {
                audio[i * row_len + j] = waveform[i][j];
            }
        }

        // batch_size * num_samples
        const int64_t batch_size = waveform.size();
        int64_t input_node_dims[2] = {batch_size, static_cast<int64_t>(waveform[0].size())};
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(audio.data()), audio.size(),
                input_node_dims, 2);
        int64_t input_node_dims1[1] = {batch_size};
        Ort::Value input_ort1 = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(wav_lens.data()), wav_lens.size(),
                input_node_dims1, 1);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(input_ort1));

        auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
                ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

        const float* outputs = ort_outputs[0].GetTensorData<float>();
        auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int len1 = outputs_shape[0];
        int len2 = outputs_shape[1];
        int len3 = outputs_shape[2];
        std::cout<<"output shape:"<<len1<<"x"<<len2<<"x"<<len3<<std::endl;

        // here len2 is always 1, len1 is batch size, len3 is always 192, which is 
        // size each embedding for each input.
        std::vector<std::vector<float>> res( data.size(), std::vector<float>( len3 ));
        for( int i = 0; i < data.size(); ++i )
        {
            for( int j = 0; j < len3; ++j )
            {
                res[i][j] = *( outputs + i * len3 + j );
            }
        }

        return res;
    }

};

class EmbeddingModel1 : public OnnxModel 
{
private:
    size_t m_batchSize = 32;


public:
    EmbeddingModel1(const std::string& model_path)
        : OnnxModel(model_path) {
    }

    /*
     * input is output of STFT
     * output: embedding
     * */
    std::vector<std::vector<float>> _infer( 
            const std::vector<std::vector<std::vector<std::vector<float>>>>& data,
            const std::vector<float>& lens )
    {
        //assert( data.size() == lens.size());
        //assert( data.size() <= m_batchSize );
        size_t d1 = data.size();
        size_t d2 = data[0].size();
        size_t d3 = data[0][0].size();
        size_t d4 = data[0][0][0].size();
        std::vector<float> wav_lens( m_batchSize, 1.0 );
        std::copy( lens.begin(), lens.end(), wav_lens.begin());

        // Create a std::vector<float> with the same size as the tensor
        size_t k = 0;
        std::vector<float> audio( m_batchSize * d2 * d3 * d4, 0.0 );
        for( size_t i = 0; i < d1; ++i )
        {
            for( size_t j = 0; j < d2; ++j )
            {
                for( size_t m = 0; m < d3; ++m )
                {
                    for( size_t n = 0; n < d4; ++n )
                    {
                        audio[k++] = data[i][j][m][n];
                    }
                }
            }
        }
        std::cout<<"input shape:"<<m_batchSize<<
            "x"<<d2<<
            "x"<<d3<<
            "x"<<d4<<std::endl;
        assert( k == d1 * d2 * d3 * d4 );
        std::ofstream f( "/tmp/sstft.txt" );
        for( size_t i = 0; i < k; ++i )
        {
            f << audio[i]<<"\n";
        }
        f.close();

        // batch_size * num_samples
        const int64_t batch_size = m_batchSize;
        int64_t input_node_dims[4] = {batch_size, 
            static_cast<int64_t>(d2), 
            static_cast<int64_t>(d3), 
            static_cast<int64_t>(d4) };
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(audio.data()), audio.size(),
                input_node_dims, 4);
        int64_t input_node_dims1[1] = {batch_size};
        Ort::Value input_ort1 = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(wav_lens.data()), wav_lens.size(),
                input_node_dims1, 1);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(input_ort1));

        auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
                ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

        const float* outputs = ort_outputs[0].GetTensorData<float>();
        auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int len1 = outputs_shape[0];
        int len2 = outputs_shape[1];
        int len3 = outputs_shape[2];
        std::cout<<"output shape:"<<len1<<"x"<<len2<<"x"<<len3<<std::endl;

        // here len2 is always 1, len1 is batch size, len3 is always 192, which is 
        // size each embedding for each input.
        std::vector<std::vector<float>> res( data.size(), std::vector<float>( len3 ));
        for( int i = 0; i < data.size(); ++i )
        {
            for( int j = 0; j < len3; ++j )
            {
                res[i][j] = *( outputs + i * len3 + j );
            }
        }

        return res;
    }

    /*
     * input: batch size x waveform
     *        wave lens
     * output: embedding
     * */
    std::vector<std::vector<float>> infer( const std::vector<std::vector<float>>& data,
            const std::vector<float>& lens )
    {
        int n_fft = 400;
        int hop_length = 160;
        int win_length = 400;

        int batch_size = data.size();
        int len = data[0].size();
        double* x = new double[batch_size * len];
        int index = 0;
        for( int i = 0; i <  batch_size; ++i )
        {
            for( int j = 0; j <  len; ++j )
            {
                *(x + index++) = data[i][j];
            }
        }
        assert( index == batch_size * len );

        torch::Tensor input = torch::from_blob(x, { batch_size, len }, torch::kFloat64);


         /*
          * Tensor stft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
                const optional<int64_t> win_lengthOpt, const c10::optional<Tensor>& window_opt,
                const bool center, c10::string_view mode, const bool normalized,
                const optional<bool> onesidedOpt, const optional<bool> return_complexOpt)
         **/
         //torch::Tensor window;
         torch::Tensor window = torch::hamming_window( win_length );
         auto y = torch::stft(input, n_fft, hop_length, win_length, window, true, "constant", false, true, false );
         //auto y = torch::stft(input.transpose(0, 1), n_fft, hop_length, win_length, window, false, true, false );
         delete []x;
         x = nullptr;

         auto s = y.transpose(2,1);
         std::cout<<s.size(0)<<std::endl;
         std::cout<<s.size(1)<<std::endl;
         std::cout<<s.size(2)<<std::endl;
         std::cout<<s.size(3)<<std::endl;
         std::vector<std::vector<std::vector<std::vector<float>>>> stft_out(
                 s.size(0), std::vector<std::vector<std::vector<float>>>( s.size(1),
                     std::vector<std::vector<float>>( s.size(2),
                         std::vector<float>( s.size(3)))));
         for( size_t i = 0; i < s.size(0); ++i )
         {
             for( size_t j = 0; j < s.size(1); ++j )
             {
                 for( size_t k = 0; k < s.size(2); ++k )
                 {
                     for( size_t m = 0; m < s.size(3); ++m )
                     {
                         //std::cout<<s[i][j][k][m].item<double>()<<",";
                         stft_out[i][j][k][m] = s[i][j][k][m].item<float>();
                     }
                 }
                 //std::cout<<std::endl;
             }
         }

         auto out = _infer( stft_out, lens );
         return out;
    }

};

class Cluster
{
private:
    // Those 2 values extracted from config.yaml under
    // ~/.cache/torch/pyannote/models--pyannote--speaker-diarization/snapshots/xxx/
    float m_threshold = 0.7153814381597874;
    size_t m_min_cluster_size = 15;

public:
    Cluster()
    {
    }

    /*
     * pyannote/audio/pipelines/clustering.py:215, __call__(...
     * embeddings: num of chunks x 3 x 192, where 192 is size each embedding
     * segmentations: num of chunks x 293 x 3, where 293 is size each segment model out[0]
     * and 3 is each segment model output[1]
     * */
    void clustering( const std::vector<std::vector<std::vector<double>>>& embeddings, 
            const std::vector<std::vector<std::vector<double>>>& segmentations, 
            std::vector<std::vector<int>>& hard_clusters, 
            int num_clusters = -1, int min_clusters = -1, int max_clusters = -1 )
    {
        // python: train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings
        std::vector<int> chunk_idx;
        std::vector<int> speaker_idx;
        auto filteredEmbeddings = filter_embeddings( embeddings, chunk_idx, speaker_idx );

#ifdef WRITE_DATA
        debugWrite2d( filteredEmbeddings, "cpp_filtered_embeddings" );
#endif // WRITE_DATA

        size_t num_embeddings = filteredEmbeddings.size();
        set_num_clusters( static_cast<int>( num_embeddings ), num_clusters, min_clusters, max_clusters );

        // do NOT apply clustering when min_clusters = max_clusters = 1
        if( max_clusters < 2 )
        {
            size_t num_chunks = embeddings.size();
            size_t num_speakers = embeddings[0].size();
            std::vector<std::vector<int>> hcluster( num_chunks, std::vector<int>( num_speakers, 0 ));
            hard_clusters.swap( hcluster );
            return;
        }

        // python: train_clusters = self.cluster(
        auto clusterRes = cluster( filteredEmbeddings, min_clusters, max_clusters, num_clusters );

#ifdef WRITE_DATA
        /*
        // for testing
        debugWrite( clusterRes, "cpp_clusterRes" );
        std::ifstream fd2("/tmp/py_clusterRes.txt"); //taking file as inputstream
        if(fd2) {
            std::ostringstream ss;
            ss << fd2.rdbuf(); // reading data
            std::string str = ss.str();
            std::string delimiter = ",";
            std::vector<std::string> v = Helper::split(str, delimiter);
            assert( v.size() - 1 == clusterRes.size());
            for( size_t i = 0; i < clusterRes.size(); ++i )
            {
                clusterRes[i] = std::stoi( v[i] );
            }
        }
        */
#endif // WRITE_DATA
        

        // python: hard_clusters, soft_clusters = self.assign_embeddings(
        assign_embeddings( embeddings, chunk_idx, speaker_idx, clusterRes, hard_clusters );
    }

    // Assign embeddings to the closest centroid
    template <typename T>
    void assign_embeddings(const std::vector<std::vector<std::vector<T>>>& embeddings,
            const std::vector<int>& chunk_idx, 
            const std::vector<int>& speaker_idx,
            const std::vector<int>& clusterRes,
            std::vector<std::vector<int>>& hard_clusters )
    {
        assert( chunk_idx.size() == speaker_idx.size());

        // python: num_clusters = np.max(train_clusters) + 1
        int num_clusters = *std::max_element(clusterRes.begin(), clusterRes.end()) + 1;
        size_t num_chunks = embeddings.size();
        size_t num_speakers = embeddings[0].size();
        size_t dimension = embeddings[0][0].size();

        // python: train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]
        std::vector<std::vector<T>> filtered_embeddings( chunk_idx.size(), 
                std::vector<T>( dimension, 0.0 ));
        for( size_t i = 0; i < chunk_idx.size(); ++i )
        {
            auto tmp = embeddings[chunk_idx[i]][speaker_idx[i]];
            for( size_t j = 0; j < dimension; ++j )
            {
                filtered_embeddings[i][j] = tmp[j];
            }
        }

        // python: centroids = np.vstack([np.mean(train_embeddings[train_clusters == k], axis=0)
        std::vector<std::vector<T>> centroids( num_clusters, std::vector<T>( dimension, 0.0 ));
        assert( filtered_embeddings.size() == clusterRes.size());
        for( int i = 0; i < num_clusters; ++i )
        {
            size_t mean_count = 0;
            for( size_t j = 0; j < clusterRes.size(); ++j )
            {
                if( i == clusterRes[j] )
                {
                    mean_count++;
                    for( size_t k = 0; k < dimension; ++k )
                    {
                        centroids[i][k] += filtered_embeddings[j][k];
                    }
                }
            }
            for( size_t k = 0; k < dimension; ++k )
            {
                centroids[i][k] /= mean_count;
            }
        }
        /*
        for( int i = 0; i < num_clusters; ++i )
        {
            for( size_t k = 0; k < dimension; ++k )
            {
                centroids[i][k] /= dimension;
            }
        }
        */

        //for k in range(num_clusters) compute distance between embeddings and clusters
        // python: rearrange(embeddings, "c s d -> (c s) d"), where d =192
        auto r1 = Helper::rearrange_down( embeddings );

        // python: cdist(
        auto dist = Helper::cosineSimilarity( r1, centroids );

#ifdef WRITE_DATA
    debugWrite2d( dist, "cpp_dist", true );
#endif // WITE_DATA
        // python: e2k_distance = rearrange(
        // N x 3 x 4 for example
        // (c s) k -> c s k
        auto soft_clusters  = Helper::rearrange_up( dist, num_chunks );

        // python: soft_clusters = 2 - e2k_distance
        for( size_t i = 0; i < soft_clusters.size(); ++i )
        {
            for( size_t j = 0; j < soft_clusters[0].size(); ++j )
            {
                for( size_t k = 0; k < soft_clusters[0][0].size(); ++k )
                {
                    soft_clusters[i][j][k] = 2.0 - soft_clusters[i][j][k];
                }
            }
        }

#ifdef WRITE_DATA
    debugWrite3d( soft_clusters, "cpp_soft_clusters", true );
#endif // WRITE_DATA

        // python: hard_clusters = np.argmax(soft_clusters, axis=2)
        //  N x 3
        hard_clusters = Helper::argmax( soft_clusters );
    }

    std::vector<std::vector<double>> filter_embeddings( 
            const std::vector<std::vector<std::vector<double>>>& embeddings, 
            std::vector<int>& chunk_idx, std::vector<int>& speaker_idx)
    {
        // **************** max_num_embeddings IS INF in python
        // Initialize vectors to store indices of non-NaN elements

        // Find non-NaN elements and store their indices
        for (int i = 0; i < embeddings.size(); ++i) {
            for (int j = 0; j < embeddings[i].size(); ++j) {
                if (!std::isnan(embeddings[i][j][0])) { // Assuming all elements in the innermost array are NaN or not NaN
                    chunk_idx.push_back(i);
                    speaker_idx.push_back(j);
                }
            }
        }

        // Sample max_num_embeddings embeddings if the number of available embeddings is greater
        /*int num_embeddings = chunk_idx.size();
        if (num_embeddings > max_num_embeddings) {
            // Shuffle the indices
            std::vector<int> indices(num_embeddings);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());

            // Sort and select the first max_num_embeddings indices
            std::sort(indices.begin(), indices.begin() + max_num_embeddings);

            // Update chunk_idx and speaker_idx with the selected indices
            chunk_idx.clear();
            speaker_idx.clear();
            for (int i = 0; i < max_num_embeddings; ++i) {
                chunk_idx.push_back(chunk_idx[indices[i]]);
                speaker_idx.push_back(speaker_idx[indices[i]]);
            }
        }*/

        // Create a vector to store the selected embeddings
        std::vector<std::vector<double>> selectedEmbeddings;
        for (int i = 0; i < chunk_idx.size(); ++i) {
            selectedEmbeddings.push_back(embeddings[chunk_idx[i]][speaker_idx[i]]);
        }

        return selectedEmbeddings;

    }

    void set_num_clusters(int num_embeddings, int& num_clusters, int& min_clusters, int& max_clusters)
    {
        if( num_clusters != -1 )
        {
            min_clusters = num_clusters;
        }
        else
        {
            if( min_clusters == -1 )
            {
                min_clusters = 1;
            }
        }
        min_clusters = std::max(1, std::min(num_embeddings, min_clusters));

        if( num_clusters != -1 )
        {
            max_clusters == num_clusters;
        }
        else
        {
            if( max_clusters == -1 )
            {
                max_clusters = num_embeddings;
            }
        }
        max_clusters = std::max(1, std::min(num_embeddings, max_clusters));
        if( min_clusters > max_clusters )
        {
            min_clusters = max_clusters;
        }
        if( min_clusters == max_clusters )
        {
            num_clusters = min_clusters;
        }
    }

    // pyannote/audio/pipelines/clustering.py:426, cluster(...
    // AgglomerativeClustering
    std::vector<int> cluster( const std::vector<std::vector<double>>& embeddings, 
            int min_clusters, int max_clusters, int num_clusters )
    {
        // python: num_embeddings, _ = embeddings.shape
        size_t num_embeddings = embeddings.size();

        // heuristic to reduce self.min_cluster_size when num_embeddings is very small
        // (0.1 value is kind of arbitrary, though)
        m_min_cluster_size = std::min( m_min_cluster_size, std::max(static_cast<size_t>( 1 ),
                    static_cast<size_t>( round(0.1 * num_embeddings))));

        // linkage function will complain when there is just one embedding to cluster
        //if( num_embeddings == 1 ) 
        //     return np.zeros((1,), dtype=np.uint8)

        // self.metric == "cosine" and self.method == "centroid"
        // python:
        //    with np.errstate(divide="ignore", invalid="ignore"):
        //        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        auto normalizedEmbeddings( embeddings );
        Helper::normalizeEmbeddings( normalizedEmbeddings );

        // python: clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        auto clusters = Clustering::cluster( normalizedEmbeddings, m_threshold );
        for( size_t i = 0; i < clusters.size(); ++i )
        {
            clusters[i] -= 1;
        }

#ifdef WRITE_DATA
    debugWrite2d( normalizedEmbeddings, "cpp_norm_embeddings" );
    debugWrite( clusters, "cpp_clusters" );
#endif // WRITE_DATA

        // split clusters into two categories based on their number of items:
        // large clusters vs. small clusters
        // python: cluster_unique, cluster_counts = np.unique(...
        std::unordered_map<int, int> clusterCountMap;
        for (int cluster : clusters) 
        {
            clusterCountMap[cluster]++;
        }

        // python: large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        // python: small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        std::vector<int> large_clusters;
        std::vector<int> small_clusters;
        for (const auto& entry : clusterCountMap) 
        {
            if ( entry.second >= m_min_cluster_size) 
            {
                large_clusters.push_back( entry.first );
            }
            else
            {
                small_clusters.push_back( entry.first );
            }
        }
        size_t num_large_clusters = large_clusters.size();

        // force num_clusters to min_clusters in case the actual number is too small
        if( num_large_clusters < min_clusters )
            num_clusters = min_clusters;

        // force num_clusters to max_clusters in case the actual number is too large
        if( num_large_clusters > max_clusters )
            num_clusters = max_clusters;

        if( num_clusters != -1 )
            assert( false ); // this branch is not implemented

        if( num_large_clusters == 0)
        {
            clusters.assign(clusters.size(),0);
            return clusters;
        }

        if( small_clusters.size() == 0 )
        {
            return clusters;
        }

        std::sort( large_clusters.begin(), large_clusters.end());
        std::sort( small_clusters.begin(), small_clusters.end());

        // re-assign each small cluster to the most similar large cluster based on their respective centroids
        auto large_centroids = Helper::calculateClusterMeans(embeddings, clusters, large_clusters);
        auto small_centroids = Helper::calculateClusterMeans(embeddings, clusters, small_clusters);

        // python: centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        auto centroids_cdist = Helper::cosineSimilarity( large_centroids, small_centroids );

        // Update clusters based on minimum distances
        // python: for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0))
        for (int small_k = 0; small_k < centroids_cdist[0].size(); ++small_k) 
        {
            float minVal = std::numeric_limits<float>::max();
            int large_k = -1;

            // np.argmin
            for (size_t i = 0; i < centroids_cdist.size(); ++i) {
                if (centroids_cdist[i][small_k] < minVal) {
                    minVal = centroids_cdist[i][small_k];
                    large_k = i;
                }
            }
            for( size_t i = 0; i < clusters.size(); ++i )
            {
                if( clusters[i] == small_clusters[small_k] )
                {
                    clusters[i] = large_clusters[large_k];
                }
            }

            std::cout << small_k << ", " << large_k << std::endl;
        }

        // Find unique clusters and return inverse mapping
        std::vector<int> uniqueClusters;
        std::vector<int> inverseMapping = Helper::findUniqueClusters(clusters, uniqueClusters);

        return inverseMapping;
    }


};

// class SpeakerDiarization{ // TODO: add this 

int embedding_batch_size = 32;
double self_frame_step = 0.016875;
double self_frame_duration = 0.016875;
double self_frame_start = 0.0;

// pyannote/audio/pipelines/speaker_verification.py:281 __call__() 
//std::vector<std::vector<double>> getEmbedding( EmbeddingModel& em, const std::vector<std::vector<float>>& dataChunks, 
std::vector<std::vector<double>> getEmbedding( EmbeddingModel1& em, const std::vector<std::vector<float>>& dataChunks, 
        const std::vector<std::vector<float>>& masks )
{
    assert( dataChunks.size() == masks.size());

    // Debug
    static int number = 0;
#ifdef WRITE_DATA
    debugWrite2d( dataChunks, std::string( "cpp_batch_waveform" ) + std::to_string( number ));
#endif // WRITE_DATA

    size_t batch_size = dataChunks.size();
    size_t num_samples = dataChunks[0].size();

    // python: imasks = F.interpolate(... ) and imasks = imasks > 0.5
    auto imasks = Helper::interpolate( masks, num_samples, 0.5 );
#ifdef WRITE_DATA
    debugWrite2d( masks, std::string("cpp_masks") + std::to_string( number ), true );
    debugWrite2d( imasks, std::string("cpp_imasks") + std::to_string( number ) );
#endif // WRITE_DATA
    
    assert( imasks.size() == batch_size );
    assert( imasks[0].size() == num_samples );
    //masks is [32x293] imask is [32x80000], dataChunks is [32x80000] as welll
    
    // python: signals = pad_sequence(...)
    auto signals = Helper::padSequence( dataChunks, imasks );
    assert( signals.size() == batch_size );
    assert( signals[0].size() == num_samples );

    // python: wav_lens = imasks.sum(dim=1)
    std::vector<float> wav_lens( batch_size, 0.0 );
    float max_len = 0;
    int index = 0;
    for( const auto& a : imasks )
    {
        float tmp = std::accumulate(a.begin(), a.end(), 0.0);
        wav_lens[index++] = tmp;
        if( tmp > max_len )
            max_len = tmp;
    }

    // python: if max_len < self.min_num_samples: return np.NAN * np.zeros(...
    if( max_len < min_num_samples )
    {
        // TODO: don't call embedding process, direct return
        // batch_size x 192, where 192 is size of length embedding result for each waveform
        // python: return np.NAN * np.zeros((batch_size, self.dimension))
        std::vector<std::vector<double>> embeddings( batch_size, std::vector<double>( 192, NAN ));
        return embeddings;
    }

#ifdef WRITE_DATA
    // Debug
    std::string fn = std::string( "cpp_wav_lens" ) + std::to_string( number );
    debugWrite( wav_lens, fn );
#endif // WRITE_DATA

    // python:         
    //      too_short = wav_lens < self.min_num_samples
    //      wav_lens = wav_lens / max_len
    //      wav_lens[too_short] = 1.0 
    std::vector<bool> too_short( wav_lens.size(), false );
    for( size_t i = 0; i < wav_lens.size(); ++i )
    {
        if( wav_lens[i] < min_num_samples )
        {
            wav_lens[i] = 1.0;
            too_short[i] = true;
        }
        else
        {
            wav_lens[i] /= max_len;
        }
    }

#ifdef WRITE_DATA
    std::string fn1 = std::string( "/tmp/cpp_final_wav_lens" ) + std::to_string( number ) + ".txt";
    debugWrite( wav_lens, fn1 );

    std::string fn2 = std::string( "/tmp/cpp_signals" ) + std::to_string( number ) + ".txt";
    debugWrite2d( signals, fn2 );
    
    number++;
#endif // WRITE_DATA

#ifdef WRITE_DATA
    /*
    debugWrite( signals[3], "wav----data" );
    debugWrite( wav_lens, "wav----len" );
    exit(0);
    */
#endif // WRITE_DATA

    // signals is [32x80000], wav_lens is of length 32 of 1d array, an example for wav_lens
    // [1.0000, 1.0000, 1.0000, 0.0512, 1.0000, 1.0000, 0.1502, ...] 
    // Now call embedding model to get embeddings of batches
    // speechbrain/pretrained/interfaces.py:903
    auto embeddings_f = em.infer( signals, wav_lens );

    // Convert float to double 
    size_t col = embeddings_f[0].size();
    std::vector<std::vector<double>> embeddings( embeddings_f.size(), 
            std::vector<double>( col ));

    // python: embeddings[too_short.cpu().numpy()] = np.NAN
    for( size_t i = 0; i < too_short.size(); ++i )
    {
        if( too_short[i] )
        {
            for( size_t j = 0; j < col; ++j )
            {
                embeddings[i][j] = NAN;
            }
        }
        else
        {
            for( size_t j = 0; j < col; ++j )
            {
                embeddings[i][j] = static_cast<double>( embeddings_f[i][j] );
            }
        }
    }
    
    return embeddings;
}

// pyannote/core/feature.py
// +
// pyannote/core/segment.py
// mode='loose', fixed=None
template<typename T>
std::vector<std::vector<T>> crop_segment( const std::vector<std::vector<T>>& data,
        const SlidingWindow& src, const Segment& focus, SlidingWindow& resFrames )
{
    size_t n_samples = data.size();
    // python: ranges = self.sliding_window.crop(
    // As we pass in Segment, so there would on range returned, here we use following
    // block code to simulate sliding_window.crop <-- TODO: maybe move following block into SlidingWindow class
    // { --> start
        // python: i_ = (focus.start - self.duration - self.start) / self.step
        float i_ = (focus.start - src.duration - src.start) / src.step;

        // python: i = int(np.ceil(i_))
        int rng_start = ceil(i_);
        if( rng_start < 0 )
            rng_start = 0;

        // find largest integer j such that
        // self.start + j x self.step <= focus.end
        float j_ = (focus.end - src.start) / src.step;
        int rng_end = floor(j_) + 1;
    // } <-- end 
    //size_t cropped_num_samples = ( rng_end - rng_start ) * m_sample_rate;
    float start = src[rng_start].start;
    SlidingWindow res( start, src.step, src.duration, n_samples );
    //auto segments = res.data();
    std::vector<Segment> segments;
    segments.push_back( Segment( rng_start, rng_end ));
    
    int n_dimensions = 1;
    // python: for start, end in ranges:
    // ***** Note, I found ranges is always 1 element returned from self.sliding_window.crop
    // if this is not true, then need change following code. Read code:
    // pyannote/core/feature.py:196
    std::vector<std::pair<int, int>> clipped_ranges;
    for( auto segment : segments )
    {
        size_t start = segment.start;
        size_t end = segment.end;

        // if all requested samples are out of bounds, skip
        if( end < 0 || start >= n_samples)
        {
            continue;
        }
        else
        {
            // keep track of non-empty clipped ranges
            // python: clipped_ranges += [[max(start, 0), min(end, n_samples)]]
            clipped_ranges.emplace_back( std::make_pair( std::max( start, 0ul ), std::min( end, n_samples )));
        }
    }
    resFrames = res;
    std::vector<std::vector<T>> cropped_data;

    // python: data = np.vstack([self.data[start:end, :] for start, end in clipped_ranges])
    for( const auto& pair : clipped_ranges )
    {
        for( int i = pair.first; i < pair.second; ++i )
        {
            std::vector<T> tmp;
            for( size_t j = 0; j < data[i].size(); ++j )
                tmp.push_back( data[i][j] );
            cropped_data.push_back( tmp );
        }
    }

    return cropped_data;
}

// pyannote/audio/pipelines/utils/diarization.py:187
bool to_diarization( std::vector<std::vector<std::vector<double>>>& segmentations, 
        const SlidingWindow& segmentations_frames,
        const std::vector<int>& count,
        const SlidingWindow& count_frames, 
        SlidingWindow& to_diarization_frames,
        std::vector<std::vector<double>>& binary)
{
    // python: activations = Inference.aggregate(...
    SlidingWindow activations_frames;
    auto activations = PipelineHelper::aggregate( segmentations, 
            segmentations_frames, 
            count_frames, 
            activations_frames, 
            false, 0.0, true );

#ifdef WRITE_DATA
    debugWrite2d( activations, "cpp_to_diarization_activations" );

    /*
    std::ifstream fd("/tmp/py_to_diarization_activations.txt"); //taking file as inputstream
    int cnt = 0;
    for( std::string line; getline( fd, line ); )
    {
        std::string delimiter = ",";
        std::vector<std::string> v = Helper::split(line, delimiter);
        v.pop_back();
        for( size_t i = 0; i < v.size(); ++i )
        {
            activations[cnt][i] = std::stof( v[i] );
        }
        cnt++;
    }
    */
#endif 

    // python: _, num_speakers = activations.data.shape
    size_t num_speakers = activations[0].size();

    // python: count.data = np.minimum(count.data, num_speakers)
    // here also convert 1d to 2d later need pass to crop_segment
    std::vector<std::vector<int>> converted_count( count.size(), std::vector<int>( 1 ));
    for( size_t i = 0; i < count.size(); ++i )
    {
        if( count[i] > num_speakers )
            converted_count[i][0] = num_speakers;
        else
            converted_count[i][0] = count[i];
    }

    // python: extent = activations.extent & count.extent
    // get extent then calc intersection, check extent() of 
    // SlidingWindowFeature and __and__() of Segment
    // Get activations.extent
    double tmpStart = activations_frames.start + (0 - .5) * activations_frames.step + 
        .5 * activations_frames.duration;
    double duration = activations.size() * activations_frames.step;
    double activations_end = tmpStart + duration;
    double activations_start = activations_frames.start;

    // Get count.extent
    tmpStart = count_frames.start + (0 - .5) * count_frames.step + .5 * count_frames.duration;
    duration = count.size() * count_frames.step;
    double count_end = tmpStart + duration;
    double count_start = count_frames.start;

    // __and__(), max of start, min of end
    double intersection_start = std::max( activations_start, count_start );
    double intersection_end = std::min( activations_end, count_end );
    Segment focus( intersection_start, intersection_end );
    SlidingWindow cropped_activations_frames;
    auto cropped_activations = crop_segment( activations, activations_frames, focus, 
            cropped_activations_frames );

    SlidingWindow cropped_count_frames;
    auto cropped_count = crop_segment( converted_count, count_frames, focus, 
            cropped_count_frames );

#ifdef WRITE_DATA
    debugWrite2d( cropped_activations, "cpp_cropped_activations" );
    debugWrite2d( cropped_count, "cpp_cropped_count" );
#endif // WRITE_DATA

    // python: sorted_speakers = np.argsort(-activations, axis=-1)
    std::vector<std::vector<int>> sorted_speakers( cropped_activations.size(),
            std::vector<int>( cropped_activations[0].size()));
    int ss_index = 0;
    for( auto& a : cropped_activations )
    {
        // -activations
        for( size_t i = 0; i < a.size(); ++i ) a[i] = -1.0 * a[i];
        auto indices = Helper::argsort( a );
        sorted_speakers[ss_index++].swap( indices );
    }
#ifdef WRITE_DATA
    debugWrite2d( sorted_speakers, "cpp_sorted_speakers" );
#endif // WRITE_DATA

    assert( cropped_activations.size() > 0 );
    assert( cropped_activations[0].size() > 0 );

    // python: binary = np.zeros_like(activations.data)
    binary.resize( cropped_activations.size(),
        std::vector<double>( cropped_activations[0].size(), 0.0 ));

    // python: for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
    // NOTE: here c is data of count, not sliding window, see __next__ of SlidingWindowFeature
    // in python code
    assert( cropped_count.size() <= sorted_speakers.size());

    // following code based on this cropped_count is one column data, if not 
    // need below code
    assert( cropped_count[0].size() == 1 );
    for( size_t i = 0; i < cropped_count.size(); ++i )
    {
        int k = cropped_count[i][0];
        assert( k <= binary[0].size());
        for( size_t j = 0; j < k; ++j )
        {
            assert( sorted_speakers[i][j] < cropped_count.size());
            binary[i][sorted_speakers[i][j]] = 1.0f;
        }
    }

    to_diarization_frames = cropped_activations_frames;

    return true;
}

// np.max( segmentation[:, cluster == k], axis=1 )
std::vector<float> max_segmentation_cluster(const std::vector<std::vector<float>>& segmentation,
                                       const std::vector<int>& cluster, int k) 
{
    std::vector<float> maxValues( segmentation.size());

    for (size_t i = 0; i < segmentation.size(); ++i) 
    {
        float maxValue = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < cluster.size(); ++j) 
        {
            if (cluster[j] == k) 
            {
                maxValue = std::max(maxValue, segmentation[i][j]);
            }
        }
        maxValues[i] = maxValue;
    }

    return maxValues;
}

// pyannote/audio/pipelines/speaker_diarization.py:403, def reconstruct(
std::vector<std::vector<double>> reconstruct( 
        const std::vector<std::vector<std::vector<float>>>& segmentations,
        const SlidingWindow& segmentations_frames,
        const std::vector<std::vector<int>>& hard_clusters, 
        const std::vector<int>& count_data,
        const SlidingWindow& count_frames,
        SlidingWindow& activations_frames)
{
    size_t num_chunks = segmentations.size();
    size_t num_frames = segmentations[0].size();
    size_t local_num_speakers = segmentations[0][0].size();

    // python: num_clusters = np.max(hard_clusters) + 1
    // Note, element in hard_clusters have negative number, don't define num_cluster as size_t
    int num_clusters = 0;
    for( size_t i = 0; i < hard_clusters.size(); ++i )
    {
        for( size_t j = 0; j < hard_clusters[0].size(); ++j )
        {
            if( hard_clusters[i][j] > num_clusters )
                num_clusters = hard_clusters[i][j];
        }
    }
    num_clusters++;
    assert( num_clusters > 0 );

    // python: for c, (cluster, (chunk, segmentation)) in enumerate(...
    std::vector<std::vector<std::vector<double>>> clusteredSegmentations( num_chunks, 
            std::vector<std::vector<double>>( num_frames, std::vector<double>( num_clusters, NAN)));
    for( size_t i = 0; i < num_chunks; ++i ) 
    {
        const auto& cluster = hard_clusters[i];
        const auto& segmentation = segmentations[i];
        for( auto k : cluster )
        {
            if( abs( k + 2 ) < std::numeric_limits<double>::epsilon()) // check if it equals -2
            {
                continue;
            }

            auto max_sc = max_segmentation_cluster( segmentation, cluster, k );
            assert( k < num_clusters );
            assert( max_sc.size() > 0 );
            assert( max_sc.size() == num_frames );
            for( size_t m = 0; m < num_frames; ++m )
            {
                clusteredSegmentations[i][m][k] = max_sc[m];
            }
        }
    }

#ifdef WRITE_DATA
    debugWrite3d( clusteredSegmentations, "cpp_clustered_segmentations" );
#endif // 

    std::vector<std::vector<double>> diarizationRes;
    to_diarization( clusteredSegmentations, segmentations_frames, 
            count_data, count_frames, activations_frames, diarizationRes );
    return diarizationRes;
}


// pyannote/audio/pipelines/utils/diarization.py:155
Annotation to_annotation( const std::vector<std::vector<double>>& scores,
        const SlidingWindow& frames,
        double onset, double offset, 
        double min_duration_on, double min_duration_off)
{
    // call binarize : pyannote/audio/utils/signal.py: 287
    size_t num_frames = scores.size();
    size_t num_classes = scores[0].size();

    // python: timestamps = [frames[i].middle for i in range(num_frames)]
    std::vector<double> timestamps( num_frames );
    for( size_t i = 0; i < num_frames; ++i )
    {
        double start = frames.start + i * frames.step;
        double end = start + frames.duration;
        timestamps[i] = ( start + end ) / 2;
    }

    // python: socre.data.T
    std::vector<std::vector<double>> inversed( num_classes, std::vector<double>( num_frames ));
    for( size_t i = 0; i < num_frames; ++i )
    {
        for( size_t j = 0; j < num_classes; ++j )
        {
            inversed[j][i] = scores[i][j];
        }
    }

    Annotation active;
    double pad_onset = 0.0;
    double pad_offset = 0.0;
    for( size_t i = 0; i< num_classes; ++i )
    {
        int label = i;
        double start = timestamps[0];
        bool is_active = false;
        if( inversed[i][0] > onset )
        {
            is_active = true;
        }
        for( size_t j = 1; j < num_frames; ++j )
        {
            // currently active
            if( is_active )
            {
                // switching from active to inactive
                if( inversed[i][j] < offset )
                {
                    Segment region(start - pad_onset, timestamps[j] + pad_offset);
                    active.addSegment(region.start, region.end, label);
                    start = timestamps[j];
                    is_active = false;
                }
            }
            else
            {
                if( inversed[i][j] > onset )
                {
                    start = timestamps[j];
                    is_active = true;
                }
            }
        }

        if( is_active )
        {
            Segment region(start - pad_onset, timestamps.back() + pad_offset);
            active.addSegment(region.start, region.end, label);
        }
    }

    // because of padding, some active regions might be overlapping: merge them.
    // also: fill same speaker gaps shorter than min_duration_off
    if( pad_offset > 0.0 || pad_onset > 0.0  || min_duration_off > 0.0 )
        active.support( min_duration_off );

    // remove tracks shorter than min_duration_on
    if( min_duration_on > 0 )
    {
        active.removeShort( min_duration_on );
    }

    return active;
}

Annotation speakerDiarization( const std::string& waveFile, const std::string& segmentModel, const std::string& embeddingModel )
{
    wav::WavReader wav_reader( waveFile );
    int num_channels = wav_reader.num_channels();
    int bits_per_sample = wav_reader.bits_per_sample();
    int sample_rate = wav_reader.sample_rate();
    const float* audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    std::vector<float> input_wav{audio, audio + num_samples};

    // Print audio samples
    for( int i = 0; i < num_samples; ++i )
    {
        input_wav[i] = input_wav[i]*1.0f/32768.0;
    }

    //*************************************
    // 1. Segmentation stage
    //*************************************
    auto beg_seg = timeNow();
    std::cout<<"\n---segmentation ---"<<std::endl;
    SegmentModel mm( segmentModel );
    std::vector<std::pair<double, double>> segments;
    SlidingWindow res_frames;
    auto segmentations = mm.slide( input_wav, res_frames );

#ifdef WRITE_DATA
    debugWrite3d( segmentations, "cpp_segmentations" );
    // for testing
    /*
    std::ifstream fd("/tmp/py_segmentations.txt"); //taking file as inputstream
    if(fd) {
        int dim1 = 0; int dim2 = 0;
        for( std::string line; getline( fd, line ); )
        {
            std::string delimiter = ",";
            std::vector<std::string> v = Helper::split(line, delimiter);
            v.pop_back();
            assert( v.size() == segmentations[0][0].size() );
            for( size_t i = 0; i < v.size(); ++i )
            {
                segmentations[dim1][dim2][i] = std::stod( v[i] );
            }
            dim2++;
            if( dim2 % segmentations[0].size()  == 0 )
            {
                dim1++;
                dim2 = 0;
            }
        }
    }
    */

#endif // WRITE_DATA

    auto segment_data = res_frames.data();
    for( auto seg : segment_data )
    {
        segments.emplace_back( std::make_pair( seg.start, seg.end ));
    }
    std::cout<<segmentations.size()<<"x"<<segmentations[0].size()<<"x"<<segmentations[0][0].size()<<std::endl;
    // estimate frame-level number of instantaneous speakers
    //std::vector<std::vector<std::vector<float>>> test = {{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}}};
    auto binarized = mm.binarize_swf( segmentations, false );
    assert( binarized.size() == segments.size());

    // estimate frame-level number of instantaneous speakers
    // In python code, binarized in speaker_count function is cacluated with 
    // same parameters as we did above, so we reuse it by passing it into speaker_count
    SlidingWindow count_frames( num_samples );
    SlidingWindow pre_frame( self_frame_start, self_frame_step, self_frame_duration );
    auto count_data = mm.speaker_count( segmentations, binarized, 
            pre_frame, count_frames, num_samples );

    // python: duration = binary_segmentations.sliding_window.duration
    double duration = 5.0;
    size_t num_chunks = binarized.size();
    size_t num_frames = binarized[0].size(); 

    // python: num_samples = duration * self._embedding.sample_rate
    size_t min_num_frames = ceil(num_frames * min_num_samples / ( duration * 16000 ));

    // python: clean_frames = 1.0 * ( np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2 )
    // python: clean_segmentations = SlidingWindowFeature( 
    //                               binary_segmentations.data * clean_frames, binary_segmentations.sliding_window )
    auto clean_segmentations = Helper::cleanSegmentations( binarized );

    assert( binarized.size() == clean_segmentations.size());
    std::vector<std::vector<float>> batchData;
    std::vector<std::vector<float>> batchMasks;

    timeCost( beg_seg, "Segmenations time" );

#ifdef WRITE_DATA
    debugWrite3d( clean_segmentations, "cpp_clean_segmentations", true );
    debugWrite3d( binarized, "cpp_binarized_segmentations" );
#endif // WRITE_DATA

    //*************************************
    // 2. Embedding
    //*************************************
    std::cout<<"\n---generating embeddings---"<<std::endl;
    auto beg_emb = timeNow();

    // Create embedding model
    //EmbeddingModel em( embeddingModel );
    EmbeddingModel1 em( embeddingModel );
    std::vector<std::vector<double>> embeddings;

    // This for loop processes python: batchify() and zip(*filter(lambda
    for( size_t i = 0; i < binarized.size(); ++i )
    {
        auto chunkData = mm.crop( input_wav, segments[i] );
        auto& masks = binarized[i];
        auto& clean_masks = clean_segmentations[i];
        assert( masks[0].size() == 3 );
        assert( clean_masks[0].size() == 3 );

        // python: for mask, clean_mask in zip(masks.T, clean_masks.T):
        for( size_t j = 0; j < clean_masks[0].size(); ++j )
        {
            std::vector<float> used_mask;
            float sum = 0.0;
            std::vector<float> reversed_clean_mask(clean_masks.size());
            std::vector<float> reversed_mask(masks.size());

            // python: np.sum(clean_mask)
            for( size_t k = 0; k < clean_masks.size(); ++k )
            {
                sum += clean_masks[k][j];
                reversed_clean_mask[k] = clean_masks[k][j];
                reversed_mask[k] = masks[k][j];
            }

            if( sum > min_num_frames )
            {
                used_mask = std::move( reversed_clean_mask );
            }
            else
            {
                used_mask = std::move( reversed_mask );
            }

            // batchify
            batchData.push_back( chunkData );
            batchMasks.push_back( std::move( used_mask ));
            if( batchData.size() == embedding_batch_size )
            {
                auto embedding = getEmbedding( em, batchData, batchMasks );
                batchData.clear();
                batchMasks.clear();

                for( auto& a : embedding )
                {
                    embeddings.push_back( std::move( a ));
                }
            }
        }
    }

    // Process remaining
    if( batchData.size() > 0 )
    {
        auto embedding = getEmbedding( em, batchData, batchMasks );
        for( auto& a : embedding )
        {
            embeddings.push_back( std::move( a ));
        }
    }

    // python: embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)
    auto embeddings1 = Helper::rearrange_up( embeddings, num_chunks );

    timeCost( beg_emb, "Embedding time" );
    
#ifdef WRITE_DATA
    /*
    // for testing - 
    // Load embeddings data generated by python to speed up
    std::ifstream fd1("/tmp/py_embeddings.txt"); //taking file as inputstream
    //std::vector<std::vector<std::vector<float>>> embeddings1(21, 
    //std::vector<std::vector<std::vector<double>>> embeddings1(109,
    std::vector<std::vector<std::vector<double>>> embeddings1(936,
            std::vector<std::vector<double>>( 3, std::vector<double>(192)));
    if(fd1) {
        int dim1 = 0; int dim2 = 0;
        int count = 0;
        for( std::string line; getline( fd1, line ); )
        {
            std::string delimiter = ",";
            std::vector<std::string> v = Helper::split(line, delimiter);
            v.pop_back();
            assert( v.size() == 192 );
            for( size_t i = 0; i < v.size(); ++i )
            {
                if( v[i] == "nan" )
                {
                    embeddings1[dim1][dim2][i] = NAN;
                }
                else
                {
                    embeddings1[dim1][dim2][i] = std::stof( v[i] );
                }
            }
            count++;
            if( count % 3 == 0 )
                dim1++;
            dim2 = count % 3;
        }
    }
    */

    debugWrite3d( embeddings1, "cpp_embeddings" );
#endif // WRITE_DATA

    //*************************************
    // 3. Clustering
    //*************************************
    // Cluster stage
    std::cout<<"\n---clustering---"<<std::endl;
    auto beg_cst = timeNow();
    Cluster cst;
    std::vector<std::vector<int>> hard_clusters; // output 1 for clustering
    cst.clustering( embeddings1, binarized, hard_clusters );

#ifdef WRITE_DATA
    debugWrite2d( hard_clusters, "cpp_hard_clusters" );
#endif // WRITE_DATA

    // keep track of inactive speakers
    //   shape: (num_chunks, num_speakers)
    // python: inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
    // python: hard_clusters[inactive_speakers] = -2
    assert( hard_clusters.size() == binarized.size());
    assert( hard_clusters[0].size() == binarized[0][0].size());
    std::vector<std::vector<float>> inactive_speakers( binarized.size(),
            std::vector<float>( binarized[0][0].size(), 0.0 ));
    for( size_t i = 0; i < binarized.size(); ++i )
    {
        for( size_t j = 0; j < binarized[0].size(); ++j )
        {
            for( size_t k = 0; k < binarized[0][0].size(); ++k )
            {
                inactive_speakers[i][k] += binarized[i][j][k];
            }
        }
    }
    for( size_t i = 0; i < inactive_speakers.size(); ++i )
    {
        for( size_t j = 0; j < inactive_speakers[0].size(); ++j )
        {
            if( abs( inactive_speakers[i][j] ) < std::numeric_limits<double>::epsilon())
                hard_clusters[i][j] = -2;
        }
    }

#ifdef WRITE_DATA
    debugWrite( count_data, "cpp_count" );
#endif // WRITE_DATA


    // python: discrete_diarization = self.reconstruct(
    // N x 4
    SlidingWindow activations_frames;
    auto discrete_diarization = reconstruct( segmentations, res_frames, 
            hard_clusters, count_data, count_frames, activations_frames );

#ifdef WRITE_DATA
    debugWrite2d( discrete_diarization, "cpp_discrete_diarization" );
#endif // WRITE_DATA

    // convert to continuous diarization
    // python: diarization = self.to_annotation(
    float diarization_segmentation_min_duration_off = 0.5817029604921046; // see SegmentModel
                                                                          
    // for testing
    /*
    std::ifstream fd("/tmp/py_discrete_diarization.txt"); //taking file as inputstream
    if(fd) {
        std::ostringstream ss;
        ss << fd.rdbuf(); // reading data
        std::string str = ss.str();
        std::string delimiter = ",";
        std::vector<std::string> v = split(str, delimiter);
        assert( v.size() - 1 == discrete_diarization.size());
        for( size_t i = 0; i < discrete_diarization.size(); ++i )
        {
            discrete_diarization[i][0] = std::stof( v[i] );
        }
    }
    */
    auto diarization = to_annotation( discrete_diarization, 
            activations_frames, 0.5, 0.5, 0.0, 
            diarization_segmentation_min_duration_off );
    timeCost( beg_cst, "Clustering time" );

    return diarization;
}

void test()
{
    /*
    std::cout<<0.5<<" - "<<Helper::np_rint( 0.5 )<<std::endl;
    std::cout<<1.1<<" - "<<Helper::np_rint( 1.2 )<<std::endl;
    std::cout<<1.5<<" - "<<Helper::np_rint( 1.5 )<<std::endl;
    std::cout<<-1.5<<" - "<<Helper::np_rint( -1.5 )<<std::endl;
    std::cout<<-2.5<<" - "<<Helper::np_rint( -2.5 )<<std::endl;
    std::cout<<-3.5<<" - "<<Helper::np_rint( -3.5 )<<std::endl;
    std::cout<<2.5<<" - "<<Helper::np_rint( 2.5 )<<std::endl;
    std::cout<<3.5<<" - "<<Helper::np_rint( 3.5 )<<std::endl;
    std::cout<<3.6<<" - "<<Helper::np_rint( 3.6 )<<std::endl;
    */

    std::cout<<"Testing closest frame"<<std::endl;
    SlidingWindow sw( 0, 0.016875, 0.016875 );
    double start = 0.0;
    std::vector<std::pair<int, float>> frames;
    for( int i = 0; i < 10000; ++i )
    {
        auto cf = sw.closest_frame( start );
        frames.emplace_back( cf, start );
        start += 0.5;
    }

    // Read expected result from file
    std::vector<std::pair<int, float>> expected_frames;
    std::ifstream fd1("../src/test/closest_frame.txt"); //taking file as inputstream
    if( fd1 ) 
    {
        for( std::string line; getline( fd1, line ); )
        {
            std::string delimiter = ",";
            std::vector<std::string> v = Helper::split(line, delimiter);
            assert( v.size() == 2 );
            expected_frames.emplace_back( std::stoi( v[0] ), std::stof( v[1] ));
        }
    }
    assert( frames == expected_frames );
    std::cout<<"==> passed"<<std::endl;

}

void testTorchScript()
{
    // testing torch script
    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = torch::jit::load("../../embeddings/embedding.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
    std::cout<<"loaded"<<std::endl;

    // Create a vector of inputs.
    std::string waveFile = "/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_1min.wav";
    wav::WavReader wav_reader( waveFile );
    const float* audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    int n = 1;
    int len = 80000;
    std::vector<float> input_wav{audio, audio + n*len};
    double* x = new double[n*len];
    for( int i = 0; i < n*len; ++i )
    {
        //input_wav[i] = input_wav[i]*1.0f/32768.0;
        *(x+i) = input_wav[i]*1.0f/32768.0;
    }
    std::vector<float> wav_lens = {1.0};
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor t = torch::from_blob(x, {n, len});
    inputs.push_back( t );
    inputs.push_back( torch::tensor( wav_lens ));
    /*
    for( int i = 0;  i < t.size(0); ++i )
    {
        for( int j = 0;  j < t.size(1); ++j )
        {
            std::cout<<t[i][j].item<double>()<<",";
        }
        std::cout<<std::endl;
    }
    */
    std::cout<<t.size(0)<<","<<t.size(1)<<","<<std::endl;
    std::cout<<"--------------"<<std::endl;


    // Execute the model and turn its output into a tensor.
    at::Tensor output = model.forward(inputs).toTensor();
    std::cout<<output.size(0)<<","<<output.size(1)<<","<<output.size(2)<<std::endl;
    for( int i = 0;  i < output.size(0); ++i )
    {
        for( int j = 0;  j < output.size(1); ++j )
        {
            for( int k = 0;  k < output.size(2); ++k )
            {
                std::cout<<output[i][j][k].item<double>()<<",";
            }
        }
        std::cout<<std::endl;
    }

}

void testSTFT( const char* modelFile )
{
    int n_fft = 400;
    int hop_length = 160;
    int win_length = 400;

    // Create a vector of inputs.
    std::string waveFile = "/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_4-speakers_Jennifer_Aniston_and_Adam_Sandler_talk.wav";
    wav::WavReader wav_reader( waveFile );
    const float* audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    int n = 32;
    int len = 80000;
    assert( n * len < num_samples );
    std::vector<float> input_wav{audio, audio + n*len};
    double* x = new double[n*len];
    for( int i = 0; i < n*len; ++i )
    {
        *(x+i) = input_wav[i]*1.0f/32768.0;
    }

    torch::Tensor input = torch::from_blob(x, { n, len }, torch::kFloat64);


     /*
      * Tensor stft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
            const optional<int64_t> win_lengthOpt, const c10::optional<Tensor>& window_opt,
            const bool center, c10::string_view mode, const bool normalized,
            const optional<bool> onesidedOpt, const optional<bool> return_complexOpt)
     **/
     torch::Tensor window = torch::hamming_window( win_length );
     //auto y = torch::stft(input3.transpose(0, 1), n_fft, hop_length, win_length, window, false, true, false );
     auto y = torch::stft(input, n_fft, hop_length, win_length, window, true, "constant", false, true, false );

     auto s = y.transpose(2,1);
     std::cout<<s.size(0)<<std::endl;
     std::cout<<s.size(1)<<std::endl;
     std::cout<<s.size(2)<<std::endl;
     std::cout<<s.size(3)<<std::endl;
     std::vector<std::vector<std::vector<std::vector<float>>>> stft_out(
             s.size(0), std::vector<std::vector<std::vector<float>>>( s.size(1),
                 std::vector<std::vector<float>>( s.size(2),
                     std::vector<float>( s.size(3)))));
     for( size_t i = 0; i < s.size(0); ++i )
     {
         for( size_t j = 0; j < s.size(1); ++j )
         {
             for( size_t k = 0; k < s.size(2); ++k )
             {
                 for( size_t m = 0; m < s.size(3); ++m )
                 {
                     //std::cout<<s[i][j][k][m].item<double>()<<",";
                     stft_out[i][j][k][m] = s[i][j][k][m].item<float>();
                 }
             }
             //std::cout<<std::endl;
         }
     }

     std::vector<float> wav_lens(n, 1.0);
     std::string mf( modelFile );
     EmbeddingModel1 em( mf );
     auto out = em._infer( stft_out, wav_lens );
     for( size_t i = 0; i < out.size(); ++i )
     {
         for( size_t j = 0; j < out[0].size(); ++j )
         {
             std::cout<<out[i][j]<<",";
         }
         std::cout<<std::endl;
     }
}

int main(int argc, char* argv[]) 
{
    //test();
    //testTorchScript();
    //testSTFT( argv[1] );
    //return 0;
    if( argc < 4 )
    {
        std::cout<<"program [segment model file] [embeding model file] [wave file]"<<std::endl;
        return 0;
    }

    auto beg = timeNow();
    std::string segmentModel( argv[1] );
    std::string embeddingModel( argv[2] );
    std::string waveFile( argv[3] );
    auto res = speakerDiarization( waveFile, segmentModel, embeddingModel );

    std::cout<<"\n----Summary----"<<std::endl;
    timeCost( beg, "Time cost" );
    std::cout<<"----------------------------------------------------"<<std::endl;
    auto diaRes = res.finalResult();
    for( const auto& dr : diaRes )
    {
        std::cout<<"["<<dr.start<<" -- "<<dr.end<<"]"<<" --> Speaker_"<<dr.label<<std::endl;
    }
    std::cout<<"----------------------------------------------------"<<std::endl;
}
