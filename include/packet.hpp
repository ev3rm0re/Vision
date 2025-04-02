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

// 序列化函数
std::vector<uint8_t> serialize_packet(const SendPacket& packet) {
    std::vector<uint8_t> buffer(sizeof(SendPacket));
    auto* ptr = reinterpret_cast<const uint8_t*>(&packet);
    buffer.assign(ptr, ptr + sizeof(SendPacket));
    
    // 计算并填充CRC校验
    crc16::Append_CRC16_Check_Sum(buffer.data(), buffer.size());
    
    // 调试输出
    // std::cout << "TX Packet: ";
    // for (auto b : buffer) {
    //     std::cout << std::hex << std::setw(2) << std::setfill('0') 
    //              << static_cast<int>(b) << " ";
    // }
    // std::cout << std::dec << "\n";
    
    return buffer;
}

// 反序列化函数
bool deserialize_packet(const uint8_t* data, size_t length, ReceivePacket& packet) {
    if (length < sizeof(ReceivePacket)) return false;
    
    // 校验CRC
    if (!crc16::Verify_CRC16_Check_Sum(data, sizeof(ReceivePacket))) {
        std::cerr << "CRC check failed\n";
        return false;
    }

    // 调试输出
    std::cout << "RX Packet: ";
    for (size_t i = 0; i < sizeof(ReceivePacket); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                 << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::dec << "\n";

    packet.header = data[0];
    packet.detect_color = data[1] & 0x01;
    packet.reset_tracker = (data[1] >> 1) & 0x01;
    packet.reserved = (data[1] >> 2) & 0x3f;
    packet.crc_checksum = (0xffff & data[sizeof(ReceivePacket) - 1]) | (data[sizeof(ReceivePacket) - 2] << 8);

    memcpy(&packet, data, sizeof(ReceivePacket));
    return true;
}

#endif