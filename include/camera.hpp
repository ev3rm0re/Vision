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
            void capture(cv::Mat* srcimg);
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
