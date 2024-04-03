#include <stream.hpp>

Stream::Stream(int initCapacity)
{
    buffer = new char[initCapacity];
    capacity = initCapacity;
    start = 0;
    length = 0;
}

Stream::~Stream()
{
    delete[] buffer;
}

int Stream::getLength()
{
    return length;
}

void Stream::append(char aChar)
{
    if (length >= capacity)
    {
        expand();
    }
    int pos = start + length;
    if (pos >= capacity)
    {
        pos -= capacity;
    }
    buffer[pos] = aChar;
    length++;
}

void Stream::append(const char *buf, int len)
{
    for (int i = 0; i < len; i++)
        append(buf[i]);
}

char Stream::peek()
{
    if (length == 0)
        return 0;
    return buffer[start];
}

int Stream::peek(char *buf, int len)
{
    if (len > length)
        len = length;
    for (int i = 0; i < len; i++)
    {
        int pos = start + i;
        if (pos >= capacity)
            pos -= capacity;
        buf[i] = buffer[pos];
    }
    return len;
}

char Stream::take()
{
    if (length == 0)
        return 0;
    char aChar = buffer[start];
    start++;
    length--;
    if (start >= capacity)
        start -= capacity;
    return aChar;
}

int Stream::take(char *buf, int len)
{
    if (len > length)
        len = length;
    for (int i = 0; i < len; i++)
        buf[i] = take();
    return len;
}

void Stream::expand()
{
    int newCapacity = capacity * 2;
    char *newBuf = new char[newCapacity];
    int newLength = length;
    take(newBuf, newLength);
    delete[] buffer;
    buffer = newBuf;
    capacity = newCapacity;
    start = 0;
    length = newLength;
}