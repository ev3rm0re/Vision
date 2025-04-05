/*
相机类定义
*/

#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <MvCameraControl.h>
#include <opencv2/opencv.hpp>

namespace HIK {
    class Camera {
        public:
            Camera(float exposureTime, float gain);
            ~Camera();
            bool open();
            void close();
            bool cap(cv::Mat* srcimg);
        private:
            int nRet;
            bool isCameraOpened;
            void* handle;
            unsigned char* pDataForBGR;
            unsigned int nPayloadSize;
            MV_CC_DEVICE_INFO* pDeviceInfo;
            MV_CC_DEVICE_INFO_LIST stDeviceList;
            MVCC_INTVALUE stParam;
            MV_FRAME_OUT stOutFrame;
            MV_CC_PIXEL_CONVERT_PARAM CvtParam;
            float exposureTime;
            float gain;
    };
} // namespace HIK

#endif // CAMERA_HPP
