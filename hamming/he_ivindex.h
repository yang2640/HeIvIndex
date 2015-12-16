/*
*  [11/19/2015] Author: Yang Zhou
*  All Rights Reserved.
*/
#ifndef HE_IVINDEX_H_INCLUDED
#define HE_IVINDEX_H_INCLUDED

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv/cv.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/flann/flann.hpp>
#include <assert.h>

using namespace std;
using namespace cv;

class Node {
public:
    Node(unsigned int id, unsigned long bits) {
        this->imgID = id;
        this->bits = bits;
    } 

    unsigned int imgID;
    unsigned long bits;
};

class IvIndex {
public:
    IvIndex(string path, Mat &codebook, size_t dbSize, Mat &proMat, Mat &medians);
    IvIndex(Mat & codebook, size_t dbSize, Mat &projMat, Mat &medians);
    ~IvIndex();

    /* add single image features into the inverted index table */
    void addToIndex(size_t imgID, string siftpath, cv::flann::Index *flannIndex);

    /* save the inverted index to disk */
    void save(string path);

    /* load the inverted index from disk */
    void load(string path);

    /* query for one image */
    vector<size_t> score(string siftpath, size_t top, cv::flann::Index *flannIndex, int threshold);

    /* batch quantization for all listed images */
    void quantize(vector<string> siftpaths, string savepath, cv::flann::Index *index);

    /* other helper functions */
    void computeIdf();
    float l2Norm(vector<size_t> &hist);
    vector<size_t> argsort(vector<float> &nums, size_t top);
    inline int hammingDist(unsigned long bit1, unsigned long bit2);
    Mat readCSV(string filename);
    void rootNormalize(Mat &descr);

    struct Comparator {
        bool operator() (pair<float, size_t> p1, pair<float, size_t> p2) {
            return p1.first > p2.first; 
        }
    } comp;

private:
    size_t codebookSz;
    size_t featureDim;
    size_t dbSize;
    /* the main inverted index structure */
    vector<vector<Node> > nodelist;
    vector<float> idf;
    vector<float> imgNorm;

    /* specific for hamming embedding */
    Mat projMat;
    Mat medians;
};
#endif
