/**********************************************************************
MIT License

Copyright (c) 2022 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors: Kiran Pamnany; Narendra Chaudhary <narendra.chaudhary@intel.com>; Sanchit Misra <sanchit.misra@intel.com>
*****************************************************************************************/

#ifndef _XSRANDOM_HPP
#define _XSRANDOM_HPP

#include <immintrin.h>
#include <utility>

class XSRandom {
public:
    XSRandom();
    unsigned next();
#if __AVX512BW__
    __m512i more();
#endif
#if ((!__AVX512BW__) & (__AVX2__))
    std::pair<__m256i, __m256i> more();
#endif
    int fillArray(unsigned int *arr, int size);

protected:
    unsigned long cong();
#if __AVX512BW__
    __m512i x, y, z, w  __attribute((aligned(64)));
#endif
#if ((!__AVX512BW__) & (__AVX2__))
    __m256i x1, x2, y1, y2, z1, z2, w1, w2  __attribute((aligned(32)));
#endif
    unsigned buffer[16];
    int index;
    unsigned long cx;
};

#endif  /* _XSRANDOM_HPP */
