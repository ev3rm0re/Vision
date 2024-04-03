/*
相机类定义
*/

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <MvCameraControl.h>
#include <opencv2/opencv.hpp>

namespace HIK
{
    class Camera
    {
        public:
            Camera();
            ~Camera();
            bool open();
            void close();
            void cap(cv::Mat* srcimg);
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
    };
} // namespace HIK

#endif // CAMERA_HPP
