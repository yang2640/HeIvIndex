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

void computeMedians(Mat& allDescriptors, string flannPath, string projMatPath, string codebookPath, string medianPath, int nBits=64, bool verbose=false) {
    if (verbose) {
        cout << "start to compute the median values for each clustering center ..." << endl;
    }
    int nData = allDescriptors.rows;

    if (verbose) {
        cout << "load the codebook ..." << endl;
    }
    /* load the codebook */
    Mat codebook;
    FileStorage fs(codebookPath, FileStorage::READ);
    fs["codebook"] >> codebook;
    fs.release();

    int nWords = codebook.rows;

    /* load the flann index */
    if (verbose) {
        cout << "load the flann index ..." << endl;
    }
    cv::flann::Index flannIndex(codebook, cv::flann::SavedIndexParams(flannPath));

    if (verbose) {
        cout << "knn search ..." << endl;
    }
    /* knn search */
    Mat flannMatchingResult;
    Mat flannMatchingDist;
    flannIndex.knnSearch(allDescriptors, flannMatchingResult, flannMatchingDist, 1, cv::flann::SearchParams(150));

    /* load the projection matrix */
    if (verbose) {
        cout << "load the projection matrix ..." << endl;
    }
    Mat projMat = readCSV(projMatPath);

    /* collect clustering assignments for each point */
    /* key: center index, value: points index */
    unordered_map<int, vector<int> > assigns;
    for (int i = 0; i < nData; ++i) {
        assigns[flannMatchingResult.at<int>(i, 0)].push_back(i);
    }

    /* compute the median values for each clustering center */
    if (verbose) {
        cout << "median computeing  ..." << endl;
    }

    Mat medians = Mat::zeros(nWords, nBits, CV_32F);
    for (int i = 0; i < nWords; ++i) {
        if (verbose && i % 5000 == 0) {
            cout << "computed medians for " << i << " centers ... ..." << endl;
        }
        Mat tmp; 
        for (size_t j = 0; j < assigns[i].size(); ++j) {
            /* i: i-th clustering center, j: j-th assign in i-th center */
            tmp.push_back(allDescriptors.row(assigns[i][j]));
        }
        tmp.push_back(codebook.row(i));
        /* project tmp with projection matrix */
        tmp = projMat * tmp.t();
        /* compute the median values of each row of tmp */
        for (int j = 0; j < nBits; ++j) {
            float* p = tmp.ptr<float>(j);
            nth_element(p, p + tmp.cols / 2, p + tmp.cols);
            medians.at<float>(i, j) = *(p + tmp.cols / 2);
        }
    }
    /* save the median matrix */
    FileStorage medianFs(medianPath, FileStorage::WRITE);
    medianFs << "median" << medians;
    medianFs.release();
}


int main() {
    string trainPath = "data/cluster/samples.sift";
    string codebookPath = "data/cluster/codebook.yml";
    string flannPath = "data/cache/flann.index";
    string projMatPath = "data/hamming/projMat.txt";
    string medianPath = "data/hamming/medians.yml";
    int nBits = 64;
    bool verbose = true;

    Mat featData = readCSV(trainPath);
    rootNormalize(featData);
    computeMedians(featData, flannPath, projMatPath, codebookPath, medianPath, nBits, verbose);
}

