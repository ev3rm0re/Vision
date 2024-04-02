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
            cv::Mat capture();
        private:
            int nRet;
            bool isCameraOpened;
            void* handle;
            unsigned char* pData;
            unsigned int nPayloadSize;
            MV_FRAME_OUT_INFO_EX stImageInfo;
            MVCC_INTVALUE stParam;
    };
} // namespace HIK
