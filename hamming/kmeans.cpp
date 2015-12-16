/*
*  [11/19/2015] Author: Yang Zhou
*  All Rights Reserved.
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <opencv/cv.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/flann/flann.hpp>
#include <assert.h>
#include <unordered_map>

using namespace std;
using namespace cv;

void rootNormalize(Mat &descr) {
    Mat L1Norm;
    reduce(descr, L1Norm, 1, CV_REDUCE_SUM);
    for (int i = 0; i < descr.rows; ++i) {
        descr.row(i) /= L1Norm.at<float>(i, 0);
    }
    sqrt(descr, descr);
}

Mat readCSV(string filename) {
    ifstream ifs(filename);
    if (!ifs.is_open()) {
        return Mat();
    }

    string line;
    Mat ret;

    while (getline(ifs, line) && !line.empty()) {
        stringstream ss(line);
        vector<float> row;
        float x = 0.0;
        while (ss >> x) {
            row.push_back(x); 
        }
        Mat tmp = Mat(row).t();
        ret.push_back(tmp); 
    }
    ifs.close();
    return ret;
}

void kmeans(Mat& allDescriptors, int nWords, string codebookPath, string flannPath, int nIter=30, bool verbose=false) {

    FileStorage fs(codebookPath, FileStorage::WRITE);
    int nData = allDescriptors.rows;
    int nDim = 128;
    Mat codebook;

    if (verbose) {
        cout << "Start kmeans: number of features: " << nData << ", feature dim: " << nDim << endl;
        cout << "number of visual words: " << nWords << endl;
        cout << "kmeans run in " << nIter << " iterations" << endl;
        cout << "-----------------------------------------------------------------" << endl;
    }

    /* init the codebook with the shuffled rows */
    vector<int> rowShuffle;
    for (int i = 0; i < nData; i++) {
        rowShuffle.push_back(i);
    }
    cv::randShuffle(rowShuffle);
    for (int i = 0; i < nWords; i++) {
        codebook.push_back(allDescriptors.row(rowShuffle[i]));
    }

    /* start the iterations */
    double clusterError = 0, lastError = 0, baseError = 0;
    for (int n = 0; n < nIter; n++) {
        if (verbose) {
            cout << "Iteration: " << n << endl;
        }

        clusterError = 0;
        Mat flannMatchingResult;
        Mat flannMatchingDist;
        cv::flann::Index flannIndex;

        if (verbose) {
            cout << "build index ..." << endl;
        }
        flannIndex.build(codebook, cv::flann::KMeansIndexParams());

        /* build index and search for the nearest neighbours */
        if (verbose) {
            cout << "knn search ..." << endl;
        }
        flannIndex.knnSearch(allDescriptors, flannMatchingResult, flannMatchingDist, 1, cv::flann::SearchParams(150));

        /* update codebook */ 
        if (verbose) {
            cout << "update clustering centers ..." << endl;
        }
        codebook = Mat::zeros(nWords, 128, CV_32F);
        vector<int> iClusterSize(nWords, 0);
        for (int i = 0; i < nData; i++) {
            codebook.row(flannMatchingResult.at<int>(i, 0)) += allDescriptors.row(i);
            iClusterSize[flannMatchingResult.at<int>(i, 0)] ++;
            clusterError += flannMatchingDist.at<float>(i, 0);
        }

        /* update error ratio*/
        if (n == 0) {
            baseError = clusterError;
        }
        else if (verbose) {
            cout << "estimated error ratio: " << abs(clusterError - lastError) / baseError << endl; 
        }
        lastError = clusterError;

        /* re-compute the codebook centers */
        for (int i = 0; i < nWords; i++) {
            if (iClusterSize[i] > 0) {
                codebook.row(i) = codebook.row(i) / iClusterSize[i];
            }
        }

        if (verbose) {
            cout << "-----------------------------------------------------------------" << endl;
        }

        if (n == nIter - 1) {
            /* re-build the index and save it, in last iteration */
            if (verbose) {
                cout << "last index build ..." << endl;
            }
            cv::flann::Index flannIndex;
            flannIndex.build(codebook, cv::flann::KMeansIndexParams());
            flannIndex.save(flannPath);
        }
    }

    /* save the codebook */
    fs << "codebook" << codebook;
    fs.release();
}

int main() {
    string trainPath = "data/cluster/samples.sift";
    string codebookPath = "data/cluster/codebook.yml";
    string flannPath = "data/cache/flann.index";
    /* define the codebook size */
    int nWords = 200000;
    int nIter = 18;
    bool verbose = true;

    Mat featData = readCSV(trainPath);
    rootNormalize(featData);
    /* last arg: verbose or not, second last arg: number of iterations for running kmeans */
    kmeans(featData, nWords, codebookPath, flannPath, nIter, verbose);
}

