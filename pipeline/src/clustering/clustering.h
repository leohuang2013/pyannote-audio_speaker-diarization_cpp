#ifndef _SPEAKER_DIARIZATION_CLUSTERING
#define _SPEAKER_DIARIZATION_CLUSTERING

class Clustering
{
public:
    static std::vector<int> cluster( const std::vector<std::vector<double>>& input, double cutoff );
    static void linkage( const std::vector<std::vector<double>>& input,
            std::vector<std::vector<double>>& dendrogram );
    static void fcluster( const std::vector<std::vector<double>>& Z, 
            double cutoff, std::vector<int>& clusters );
};

#endif // _SPEAKER_DIARIZATION_CLUSTERING
