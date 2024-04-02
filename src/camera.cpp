﻿#include "camera.hpp"

namespace HIK
{
    Camera::Camera()
    {
        nRet = MV_OK;
        isCameraOpened = false;
        handle = NULL;
        nPayloadSize = 0;
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        CvtParam = {0};
        stOutFrame = {0};
        memset(&stOutFrame, 0, sizeof(MV_FRAME_OUT));
    }

    Camera::~Camera()
    {
        if (isCameraOpened)
        {
            close();
        }
        if (handle != NULL)
        {
            MV_CC_DestroyHandle(handle);
        }
    }

    bool Camera::open()
    {
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_EnumDevices fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
        if (stDeviceList.nDeviceNum > 0)
        {
            for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
            {
                std::cout << "Device: " << i << std::endl;
            }
        }
        else
        {
            std::cerr << "No device found!" << std::endl;
            return false;
        }

        // 选择相机
        unsigned int nIndex = 0;
        nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_CreateHandle fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 打开相机
        nRet = MV_CC_OpenDevice(handle);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_OpenDevice fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 关闭触发模式
        nRet = MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_SetEnumValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 设置图像格式
        nRet = MV_CC_SetEnumValue(handle, "PixelFormat", 0x0210001F);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_SetEnumValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
        // 设置曝光时间
        nRet = MV_CC_SetFloatValue(handle, "ExposureTime", 10000);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_SetFloatValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
        // 设置增益
        nRet = MV_CC_SetFloatValue(handle, "Gain", 8.0);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_SetFloatValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }

        // 获取图像大小
        nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_GetIntValue fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        }
        nPayloadSize = stParam.nCurValue;

        // 开始取流
        nRet = MV_CC_StartGrabbing(handle);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_StartGrabbing fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
            return false;
        }
        return true;
    }

    void Camera::capture(cv::Mat* srcimg)
    {
        nRet = MV_CC_GetImageBuffer(handle, &stOutFrame, 400);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_GetImageBuffer fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        }
        CvtParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType;
        CvtParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
        CvtParam.nWidth = stOutFrame.stFrameInfo.nWidth;
        CvtParam.nHeight = stOutFrame.stFrameInfo.nHeight;
        pDataForBGR = (unsigned char*)malloc(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 4 + 2048);
        CvtParam.pSrcData = stOutFrame.pBufAddr;
        CvtParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen;
        CvtParam.nDstBufferSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 4 + 2048;
        CvtParam.pDstBuffer = pDataForBGR;
        nRet = MV_CC_ConvertPixelType(handle, &CvtParam);
        if (MV_OK != nRet)
        {
            std::cerr << "MV_CC_ConvertPixelType fail! nRet [0x" << std::hex << nRet << "]" << std::endl;
        }
        *srcimg = cv::Mat(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, CV_8UC3, pDataForBGR);
        if (NULL != stOutFrame.pBufAddr)
        {
            MV_CC_FreeImageBuffer(handle, &stOutFrame);
        }
    }

    void Camera::close()
    {
        MV_CC_StopGrabbing(handle);
        MV_CC_CloseDevice(handle);
    }
}