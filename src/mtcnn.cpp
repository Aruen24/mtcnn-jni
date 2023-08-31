/*
created by L. 2018.05.16
*/

#include"mtcnn.h"
#include <algorithm>
#include <map>

mtcnn::mtcnn(const std::string &model_path, int min_size, int thread_num) {

    Pnet.load_param((model_path + "/det1.param").data());
    Pnet.load_model((model_path + "/det1.bin").data());
    Rnet.load_param((model_path + "/det2.param").data());
    Rnet.load_model((model_path + "/det2.bin").data());
    Onet.load_param((model_path + "/det3.param").data());
    Onet.load_model((model_path + "/det3.bin").data());
    this->min_size = min_size;
    this->threadnum = thread_num;
}

mtcnn::~mtcnn() {
    Pnet.clear();
    Rnet.clear();
    Onet.clear();
}
void mtcnn::setMinFace(int minSize) {
    min_size = minSize;
}
void mtcnn::setThreadNum(int threadNum) {
    threadnum=threadNum;
}
void mtcnn::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, float scale) {
    const int stride = 2;
    const int cellsize = 12;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
    //float *plocal = location.data;
    Bbox bbox;
    float inv_scale = 1.0f / scale;
    for (int row = 0; row<score.h; row++) {
        for (int col = 0; col<score.w; col++) {
            if (*p>threshold[0]) {
                bbox.score = *p;
                bbox.x1 = round((stride*col + 1)*inv_scale);
                bbox.y1 = round((stride*row + 1)*inv_scale);
                bbox.x2 = round((stride*col + 1 + cellsize)*inv_scale);
                bbox.y2 = round((stride*row + 1 + cellsize)*inv_scale);
                bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
                const int index = row * score.w + col;
                for (int channel = 0; channel<4; channel++) {
                    bbox.regreCoord[channel] = location.channel(channel)[index];
                }
                boundingBox_.push_back(bbox);
            }
            p++;
            //plocal++;
        }
    }
}
void mtcnn::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, std::string modelname) {
    if (boundingBox_.empty()) {
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    std::vector<int> vPick;
    int nPick = 0;
    std::multimap<float, int> vScores;
    const int num_boxes = boundingBox_.size();
    vPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i) {
        vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
    }
    while (vScores.size() > 0) {
        int last = vScores.rbegin()->second;
        vPick[nPick] = last;
        nPick += 1;
        for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
            int it_idx = it->second;
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //maxX1 and maxY1 reuse
            maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if (!modelname.compare("Union"))
                IOU = IOU / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
            else if (!modelname.compare("Min")) {
                IOU = IOU / ((boundingBox_.at(it_idx).area < boundingBox_.at(last).area) ? boundingBox_.at(it_idx).area : boundingBox_.at(last).area);
            }
            if (IOU > overlap_threshold) {
                std::multimap<float, int>::iterator next_it = it;
                next_it++;
                //it = vScores.erase(it);
                vScores.erase(it);
                it = next_it;
            }
            else {
                it++;
            }
        }
    }

    vPick.resize(nPick);
    std::vector<Bbox> tmp_;
    tmp_.resize(nPick);
    for (int i = 0; i < nPick; i++) {
        tmp_[i] = boundingBox_[vPick[i]];
    }
    boundingBox_ = tmp_;
}
void mtcnn::refine(std::vector<Bbox> &vecBbox, const int &height, const int &width, bool square) {
    if (vecBbox.empty()) {
        std::cout << "Bbox is empty!!" << std::endl;
        return;
    }
    float bbw = 0, bbh = 0, maxSide = 0;
    float h = 0, w = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    for (std::vector<Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++) {
        bbw = (*it).x2 - (*it).x1 + 1;
        bbh = (*it).y2 - (*it).y1 + 1;
        x1 = (*it).x1 + (*it).regreCoord[0] * bbw;
        y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
        x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
        y2 = (*it).y2 + (*it).regreCoord[3] * bbh;

        if (square) {
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h>w) ? h : w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);
        }

        //boundary check
        if ((*it).x1<0)(*it).x1 = 0;
        if ((*it).y1<0)(*it).y1 = 0;
        if ((*it).x2>width)(*it).x2 = width - 1;
        if ((*it).y2>height)(*it).y2 = height - 1;

        it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
    }
}
void mtcnn::PNet() {
    firstBbox_.clear();
    float minl = img_w < img_h ? img_w : img_h;
    float m = (float)MIN_DET_SIZE / min_size;
    minl *= m;
    float factor = pre_facetor;
    std::vector<float> scales_;
    while (minl>MIN_DET_SIZE) {
        scales_.push_back(m);
        minl *= factor;
        m = m*factor;
    }

    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        ncnn::Mat in;
        resize_bilinear(img, in, ws, hs);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_num_threads(threadnum);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        std::vector<Bbox> boundingBox_;
        generateBbox(score_, location_, boundingBox_, scales_[i]);
        nms(boundingBox_, nms_threshold[0]);
        firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
        boundingBox_.clear();
    }
}
void mtcnn::RNet() {
    secondBbox_.clear();
    int count = 0;
    for (std::vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++) {
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 24, 24);
        ncnn::Extractor ex = Rnet.create_extractor();
        ex.set_num_threads(threadnum);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox;
        ex.extract("prob1", score);
        ex.extract("conv5-2", bbox);
        if ((float)score[1] > threshold[1]) {
            for (int channel = 0; channel<4; channel++) {
                it->regreCoord[channel] = (float)bbox[channel];//*(bbox.data+channel*bbox.cstep);
            }
            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
            it->score = score.channel(1)[0];//*(score.data+score.cstep);
            secondBbox_.push_back(*it);
        }
    }
}
void mtcnn::ONet() {
    thirdBbox_.clear();
    for (std::vector<Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++) {
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 48, 48);
        ncnn::Extractor ex = Onet.create_extractor();
        ex.set_num_threads(threadnum);
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox, keyPoint;
        ex.extract("prob1", score);
        ex.extract("conv6-2", bbox);
        ex.extract("conv6-3", keyPoint);
        if ((float)score[1] > threshold[2]) {
            for (int channel = 0; channel < 4; channel++) {
                it->regreCoord[channel] = (float)bbox[channel];
            }
            it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
            it->score = score[1];
            for (int num = 0; num<5; num++) {
                (it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * keyPoint[num];
                (it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1) * keyPoint[num + 5];
            }

            thirdBbox_.push_back(*it);
        }
    }
}

void mtcnn::detect(const ncnn::Mat& ncnn_img, std::vector<Bbox>& finalBbox_) {
    img = ncnn_img.clone();
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

    PNet();
    //the first stage's nms
    if (firstBbox_.size() < 1) return;
    nms(firstBbox_, nms_threshold[0]);
    refine(firstBbox_, img_h, img_w, true);

    //second stage
    RNet();
    if (secondBbox_.size() < 1) return;
    nms(secondBbox_, nms_threshold[1]);
    refine(secondBbox_, img_h, img_w, true);

    //third stage
    ONet();
    if (thirdBbox_.size() < 1) return;
    refine(thirdBbox_, img_h, img_w, true);
    nms(thirdBbox_, nms_threshold[2], "Min");

    finalBbox_.clear();
    for(int i=0; i<thirdBbox_.size(); ++i) {
        Bbox box(thirdBbox_[i]);
        finalBbox_.push_back(box);
    }
}


