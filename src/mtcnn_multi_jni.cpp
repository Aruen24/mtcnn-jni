//
// Created by hutian on 2019/11/7.
//

#include <jni.h>
#include <string>
#include <vector>

#include "ncnn/net.h"
#include "mtcnn.h"

//static mtcnn *mMtcnn;
std::vector< mtcnn* > vec_mtcnn;
bool detection_init_ok = false;

extern "C" {
JNIEXPORT jboolean JNICALL
Java_mtcnn_Mtcnn_initDetect(JNIEnv *env, jobject instance, jstring detectModelPath_,
                        jint minDetSize, jint num) {
    if(detection_init_ok) {
        return true;
    }
    if (NULL == detectModelPath_) {
        return false;
    }

    const char *detectModelPath = env->GetStringUTFChars(detectModelPath_, 0);
    if (NULL == detectModelPath) {
        return false;
    }

    std::string tModelDir = detectModelPath;
    std::string tLastChar = tModelDir.substr(tModelDir.length() - 1, 1);
    // 目录补齐/
    if ("\\" == tLastChar) {
        tModelDir = tModelDir.substr(0, tModelDir.length() - 1) + "/";
    } else if (tLastChar != "/") {
        tModelDir += "/";
    }

    for(int i=0; i<num; ++i) {
        vec_mtcnn.push_back(new mtcnn(tModelDir, minDetSize, 1));
    }

    env->ReleaseStringUTFChars(detectModelPath_, detectModelPath);

    detection_init_ok = true;

    return true;
}

JNIEXPORT jobjectArray JNICALL
Java_mtcnn_Mtcnn_detect(JNIEnv *env, jobject instance, jbyteArray frame_, jint width_, jint height_, jint id) {
    jbyte *frame = env->GetByteArrayElements(frame_, NULL);
    unsigned char *faceImageCharData = (unsigned char *) frame;

    ncnn::Mat ncnn_img;
    ncnn_img = ncnn::Mat::from_pixels(faceImageCharData, ncnn::Mat::PIXEL_RGB,
                                      width_, height_);

    std::vector<Bbox> boxes;
    vec_mtcnn[id]->detect(ncnn_img, boxes);

    env->ReleaseByteArrayElements(frame_, frame, 0);

    jobjectArray ret = env->NewObjectArray(boxes.size(), env->FindClass("[F"), NULL);
    for(int i=0; i<boxes.size(); ++i) {
        jfloatArray iarr = env->NewFloatArray(15);
        jfloat tmp[15];
        tmp[0] = (float)boxes[i].x1;
        tmp[1] = (float)boxes[i].y1;
        tmp[2] = (float)boxes[i].x2;
        tmp[3] = (float)boxes[i].y2;
        tmp[4] = boxes[i].ppoint[0];
        tmp[5] = boxes[i].ppoint[1];
        tmp[6] = boxes[i].ppoint[2];
        tmp[7] = boxes[i].ppoint[3];
        tmp[8] = boxes[i].ppoint[4];
        tmp[9] = boxes[i].ppoint[5];
        tmp[10] = boxes[i].ppoint[6];
        tmp[11] = boxes[i].ppoint[7];
        tmp[12] = boxes[i].ppoint[8];
        tmp[13] = boxes[i].ppoint[9];
        tmp[14] = boxes[i].score;
        env->SetFloatArrayRegion(iarr, 0, 15, tmp);
        env->SetObjectArrayElement(ret, i, iarr);
        env->DeleteLocalRef(iarr);
    }

    return ret;
}

JNIEXPORT void JNICALL
Java_mtcnn_Mtcnn_setMinSize(JNIEnv *env, jobject instance, jint minSize_, jint id) {
    vec_mtcnn[id]->setMinFace(minSize_);
}

JNIEXPORT void JNICALL
Java_mtcnn_Mtcnn_releaseDetect(JNIEnv *env, jobject instance) {
    if (!detection_init_ok) {
        return;
    }

    for(int i=0; i<vec_mtcnn.size(); ++i) {
        delete vec_mtcnn[i];
    }

    vec_mtcnn.erase(vec_mtcnn.begin(), vec_mtcnn.end());

    detection_init_ok = false;
}
}

