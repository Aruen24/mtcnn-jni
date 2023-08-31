#pragma once
#ifndef _MTCNN_H_
#define _MTCNN_H_
#include <string>
#include <iostream>
#include "ncnn/net.h"

struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    float ppoint[10];
    float regreCoord[4];

    Bbox() {}

    Bbox(const Bbox& box) {
        score = box.score;
        x1 = box.x1;
        y1 = box.y1;
        x2 = box.x2;
        y2 = box.y2;
        area = box.area;
        for(int i=0; i<10; ++i) {
            ppoint[i] = box.ppoint[i];
        }
        for(int i=0; i<4; ++i) {
            regreCoord[i] = box.regreCoord[i];
        }
    }
};

class mtcnn {
    public:
        mtcnn(const std::string &model_path, int min_size, int thread_num);
        mtcnn(const std::vector<std::string> param_files, const std::vector<std::string> bin_files);
        ~mtcnn();
        void setMinFace(int minSize);
        void setThreadNum(int threadNum);

        void detect(const ncnn::Mat& img, std::vector<Bbox>& finalBbox);
        static bool cmpScore(Bbox lsh, Bbox rsh) {
            if (lsh.score < rsh.score)
                return true;
            else
                return false;
        }
private:
        void generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, float scale);
        void nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, std::string modelname = "Union");
        void refine(std::vector<Bbox> &vecBbox, const int &height, const int &width, bool square);
        void PNet();
        void RNet();
        void ONet();
        ncnn::Net Pnet, Rnet, Onet;
        ncnn::Mat img;
        const float nms_threshold[3] = { 0.5f, 0.7f, 0.7f };

        const float mean_vals[3] = { 127.5, 127.5, 127.5 };
        const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
        const int MIN_DET_SIZE = 12;
        std::vector<Bbox> firstBbox_, secondBbox_, thirdBbox_;
        int img_w, img_h;

    private:
        const float threshold[3] = { 0.7f, 0.7f, 0.7f };
        const float pre_facetor = 0.709f;
        int threadnum = 1;
        int min_size;
    };
#endif // !DETECT_H_
