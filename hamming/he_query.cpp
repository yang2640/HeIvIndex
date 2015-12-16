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

inline bool exists(string name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

int main() {
    /* load codebook */
    string codebookPath = "data/cluster/codebook.yml";
    Mat codebook;
    FileStorage codebookFs(codebookPath, FileStorage::READ);
    codebookFs["codebook"] >> codebook;
    codebookFs.release();

    /* load flann */
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
    CvMLData projMatData;
    projMatData.read_csv(projMatPath.c_str());
    Mat projMat(projMatData.get_values());

    /* load median */
    string medianPath = "data/hamming/medians.yml";
    Mat medians;
    FileStorage fs(medianPath, FileStorage::READ);
    fs["median"] >> medians;
    fs.release();

    /* query */
    vector<string> siftpaths = readlines("data/featlist");
    // only take 2347 - 3347 for debug
    vector<string> tmp;
    for (int i = 0; i < 10200; ++i) {
        tmp.push_back(siftpaths[i]);
    }
    string savepath = "data/cache/ivindex.txt";
    int threshold = 26;
    if (exists(savepath)) {
        IvIndex ivindex(savepath, codebook, tmp.size(), projMat, medians);
        for (size_t i = 0; i < tmp.size(); ++i) {
            vector<size_t> ret = ivindex.score(tmp[i], 15, flannIndex, threshold);
            for (auto x: ret) {
                cout << x << " ";
            }
            cout << endl;
        }
    }
    else {
        cout << "index file doesn't exist ... ..." << endl;
    }

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
