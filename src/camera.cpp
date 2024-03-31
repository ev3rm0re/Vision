#include "camera.hpp"
#include <iostream>
#include <string.h>

namespace HIK
{
    Camera::Camera()
    {
        m_hDevHandle = NULL;
        stOutFrame = {0};
        memset(&stOutFrame, 0, sizeof(MV_FRAME_OUT));
    }

    Camera::~Camera()
    {
        if (m_hDevHandle)
        {
            MV_CC_DestroyHandle(m_hDevHandle);
            m_hDevHandle = 0;
        }
    }

    void Camera::open()
    {
        MV_CC_DEVICE_INFO_LIST stDevList;
        memset(&stDevList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDevList);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_EnumDevices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return;
        }
        if (stDevList.nDeviceNum == 0)
        {
            std::cerr << "Find no device!" << std::endl;
            return;
        }
        nRet = MV_CC_CreateHandle(&m_hDevHandle, stDevList.pDeviceInfo[0]);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_CreateHandle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return;
        }
        nRet = MV_CC_OpenDevice(m_hDevHandle);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_OpenDevice fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return;
        }
        int nRet_Format = MV_CC_SetEnumValue(m_hDevHandle, "PixelFormat", PixelType_Gvsp_RGB8_Packed);
        int nRet_ExA = MV_CC_SetEnumValue(m_hDevHandle, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
        int nRet_ExT = MV_CC_SetFloatValue(m_hDevHandle, "ExposureTime", 5000);
        int nRet_Gain = MV_CC_SetFloatValue(m_hDevHandle, "Gain", 8.0);
        if (MV_OK != nRet_Format || MV_OK != nRet_ExA || MV_OK != nRet_ExT || MV_OK != nRet_Gain)
        {
            std::cerr << "MV_CC_SetValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return;
        }
        nRet = MV_CC_StartGrabbing(m_hDevHandle);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_StartGrabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return;
        }
    }

    cv::Mat Camera::capture()
    {
        MV_CC_GetImageBuffer(m_hDevHandle, &stOutFrame, 1000);
        cv::Mat frame(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, CV_8UC3, stOutFrame.pBufAddr);
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        MV_CC_FreeImageBuffer(m_hDevHandle, &stOutFrame);
        return frame;
    }
}
