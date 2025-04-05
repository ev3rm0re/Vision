#ifndef _PACKET_HPP_
#define _PACKET_HPP_

#include <boost/asio.hpp>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <crc.hpp>

#pragma pack(push, 1)
struct SendPacket {
    uint8_t header = 0xA5;
    bool tracking : 1;
    uint8_t id : 3;
    uint8_t armors_num : 3; // 2-balance 3-outpost 4-normal
    uint8_t reserved : 1;
    float yaw;
    float pitch;
    float distance;
    uint8_t end = 0x0D;
    uint8_t newline = 0x0A;
    uint16_t crc_checksum = 0;
};

struct ReceivePacket {
    uint8_t header = 0x5A;
    uint8_t detect_color : 1; // 0-red 1-blue
    bool reset_tracker : 1;
    uint8_t reserved : 6;
    uint16_t crc_checksum = 0;
};
#pragma pack(pop)

#endif