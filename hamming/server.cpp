#include "ivindex.h"
#include <sys/stat.h>
#include <unistd.h>
#include <zmq.hpp>
#include <zhelpers.hpp>

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
    // load codebook
    cout << "load codebook ... ..." << endl;
    string cbpath = "data/cluster/clst.npy";
    CvMLData mlData;
    mlData.read_csv(cbpath.c_str());
    Mat codebook(mlData.get_values());

    // load flann
    cout << "build flann ... ... " << endl;
    string flannpath = "data/cache/flann.index";
    cv::flann::IndexParams *indexParams;
    cv::flann::Index *flannIndex;
    if (exists(flannpath)) {
        indexParams = new cv::flann::SavedIndexParams(flannpath);
        flannIndex = new cv::flann::Index(codebook, *indexParams);
    }
    else {
        indexParams = new cv::flann::AutotunedIndexParams(); 
        flannIndex = new cv::flann::Index(codebook, *indexParams);
        flannIndex->save(flannpath);
    }

    // create the inverted index
    vector<string> siftpaths = readlines("data/featlist");
    string savepath = "data/cache/ivindex.txt";
    if (!exists(savepath)) {
        cerr << "index file doesn't exist ... ..." << endl;
        exit(-1);
    }
    IvIndex ivindex(savepath, codebook, siftpaths.size());

    // prepare the context and sockets
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");

    cout << "start to receive request ... ..." << endl;
    // query 
    while (true) {
        // Wait for next request from client
        string recvStr = s_recv(socket);
        // ???? transform recvStr ukbench00000.th.jpg to data/Images/ukbench00000.jpg.sift
        char path[128];
        sprintf(path, "data/Images/ukbench%s.jpg.sift", recvStr.substr(7, 5).c_str());
        cout << "process request: " << recvStr << endl;
        vector<size_t> ret = ivindex.score(path, 15, flannIndex);

        sleep(1);
        // Send reply back to client
        string reply;
        for (auto x: ret) {
            reply += to_string(x) + " ";
        }
        s_send(socket, reply);
    }

    delete indexParams;
    indexParams = nullptr;
    delete flannIndex;
    flannIndex = nullptr;

    return 0;
}
