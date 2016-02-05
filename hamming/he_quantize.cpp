/*
*  [11/19/2015] Author: Yang Zhou
*  All Rights Reserved.
*/

#include "he_ivindex.h"
#include <sys/stat.h>

vector<string> readlines(string path) {
    ifstream ifs(path);
    vector<string> lines;
    if (ifs.is_open()) {
        string line;
        while (getline(ifs, line)) {
            if (line.empty()) {
                continue;
            } 
            lines.push_back(line);
        } 
        ifs.close();
    }
    else {
        cerr << "error: can't open file " << path << endl;
    }
    return lines;
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

inline bool exists(string name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

int main() {
    /* load codebook */
    cout << "load codebook ... ..." << endl;
    string codebookPath = "data/cluster/codebook.yml";
    Mat codebook;
    FileStorage codebookFs(codebookPath, FileStorage::READ);
    codebookFs["codebook"] >> codebook;
    codebookFs.release();

    /* load flann */
    cout << "build flann ... ... " << endl;
    string flannpath = "data/cache/flann.index";
    cv::flann::IndexParams *indexParams = nullptr;
    cv::flann::Index *flannIndex = nullptr;
    if (exists(flannpath)) {
        indexParams = new cv::flann::SavedIndexParams(flannpath);
        flannIndex = new cv::flann::Index(codebook, *indexParams);
    }
    else {
        indexParams = new cv::flann::KMeansIndexParams(); 
        flannIndex = new cv::flann::Index(codebook, *indexParams);
        flannIndex->save(flannpath);
    }

    /* load projection matrix */
    string projMatPath = "data/hamming/projMat.txt";
    Mat projMat = readCSV(projMatPath);

    /* load median */
    string medianPath = "data/hamming/medians.yml";
    Mat medians;
    FileStorage fs(medianPath, FileStorage::READ);
    fs["median"] >> medians;
    fs.release();

    /* quantize */
    vector<string> siftpaths = readlines("data/featlist");
    string savepath = "data/cache/ivindex.txt";

    IvIndex ivindex(codebook, siftpaths.size(), projMat, medians);
    ivindex.quantize(siftpaths, savepath, flannIndex);

    if (indexParams != nullptr) {
        delete indexParams;
        indexParams = nullptr;
    }
    if (flannIndex != nullptr) {
        delete flannIndex;
        flannIndex = nullptr;
    }
    return 0;
}
