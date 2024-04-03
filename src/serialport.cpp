/*
串口主模块
*/

#include <timestamp.hpp>
#include <serialport.hpp>

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

Serial::Serial() : stream()
{
    pthread_mutex_init(&mutex, 0);
}

Serial::~Serial()
{
    close();
}

void *receiveThread(void *arg)
{
    Serial *serial = (Serial *)arg;
    char buf[1024];
    while (true)
    {
        pthread_testcancel();
        int len = read(serial->fd, buf, sizeof(buf));
        if (len > 0)
        {
            pthread_mutex_lock(&(serial->mutex));
            serial->stream.append(buf, len);
            pthread_mutex_unlock(&(serial->mutex));
        }
        usleep(1000);
    }
}

int Serial::open(const char *dev, int baud, int dataBits, int parityMode, int stopBits)
{
    struct termios options;
    bzero(&options, sizeof(options));
    int baudT = transformBaud(baud);
    if (baudT < 0)
        return BAUD_NOT_SUPPORTED;
    cfsetispeed(&options, baudT);
    cfsetospeed(&options, baudT);
    int dataBitsT = transformDataBits(dataBits);
    if (dataBitsT < 0)
        return DATABITS_NOT_SUPPORTED;
    options.c_cflag |= dataBitsT;
    if (parityMode == PARITY_ODD)
    {
        options.c_cflag |= PARENB;
        options.c_cflag |= PARODD;
    }
    else if (parityMode == PARITY_EVEN)
        options.c_cflag |= PARENB;
    else if (parityMode != PARITY_NONE)
        return PARITYMODE_NOT_SUPPORTED;
    if (stopBits == 2)
        options.c_cflag |= CSTOPB;
    else if (stopBits != 1)
        return STOPBITS_NOT_SUPPORTED;
    options.c_cc[VTIME] = 1;
    options.c_cc[VMIN] = 1;
    fd = ::open(dev, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0)
        return DEV_NOT_FOUND;
    if (tcsetattr(fd, TCSANOW, &options))
        return CONFIG_FAIL;
    if (tcflush(fd, TCIOFLUSH))
        return CONFIG_FAIL;
    if (pthread_create(&tid, 0, receiveThread, this) != 0)
        return NEW_THREAD_FAIL;
    return OK;
}

void Serial::close()
{
    if (fd >= 0)
    {
        ::close(fd);
        pthread_cancel(tid);
        pthread_mutex_destroy(&mutex);
        fd = -1;
    }
}

int Serial::write(const char *data, int len)
{
    return ::write(fd, data, len);
}

int Serial::available()
{
    int len = stream.getLength();
    return len;
}

int Serial::peek(char *buf, int len)
{
    len = stream.peek(buf, len);
    return len;
}

int Serial::read(char *buf, int len, int timeout)
{
    timestamp_t start = Timestamp::now();
    int total = 0;
    while (total < len)
    {
        pthread_mutex_lock(&mutex);
        int readLen = stream.take(buf + total, len - total);
        pthread_mutex_unlock(&mutex);
        if (readLen > 0)
            total += readLen;
        timestamp_t now = Timestamp::now();
        if (now >= start + timeout)
            break;
        usleep(1000);
    }
    return total;
}

int Serial::read(char *buf, int maxLen, const char *end, int timeout, int *recvLen)
{
    int endLen = strlen(end);
    timestamp_t start = Timestamp::now();
    int total = 0;
    while (total < maxLen)
    {
        pthread_mutex_lock(&mutex);
        int readLen = stream.take(buf + total, 1);
        pthread_mutex_unlock(&mutex);
        if (readLen > 0)
        {
            total += readLen;
            if (endsWith(buf, total, end, endLen))
            {
                if (recvLen != 0)
                    *recvLen = total;
                return READ_END;
            }
        }
        timestamp_t now = Timestamp::now();
        if (now >= start + timeout)
            return READ_TIMEOUT;
        usleep(1000);
    }
    return READ_BUFFER_FULL;
}

int Serial::transformBaud(int baud)
{
    int map[][2] = {{2400, B2400}, {4800, B4800}, {9600, B9600}, {19200, B19200}, {38400, B38400}, {57600, B57600}, {115200, B115200}};
    for (int i = 0; i < sizeof(map) / sizeof(int) / 2; i++)
        if (map[i][0] == baud)
            return map[i][1];
    return -1;
}

int Serial::transformDataBits(int dataBits)
{
    int map[][2] = {{5, CS5}, {6, CS6}, {7, CS7}, {8, CS8}};
    for (int i = 0; i < sizeof(map) / sizeof(int) / 2; i++)
        if (map[i][0] == dataBits)
            return map[i][1];
    return -1;
}

bool Serial::endsWith(const char *str, int strLen, const char *end, int endLen)
{
    if (strLen < endLen)
        return false;
    for (int i = endLen - 1; i >= 0; i--)
        if (end[i] != str[strLen - endLen + i])
            return false;
    return true;
}