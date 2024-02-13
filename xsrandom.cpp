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



#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <x86intrin.h>
#include "xsrandom.hpp"
#include <utility>

XSRandom::XSRandom()
{
    int seeds[64]  __attribute((aligned(64)));
    cx = (unsigned long)_rdtsc();

    for (int i = 0;  i < 64;  i++)
        seeds[i] = (int)cong();

#if __AVX512BW__
    x = _mm512_load_epi32(&seeds[0]);
    y = _mm512_load_epi32(&seeds[16]);
    z = _mm512_load_epi32(&seeds[32]);
    w = _mm512_load_epi32(&seeds[48]);
#endif
#if ((!__AVX512BW__) & (__AVX2__))
    x1 = _mm256_load_si256  ((const __m256i*)&seeds[0]);
    x2 = _mm256_load_si256  ((const __m256i*)&seeds[8]);
    y1 = _mm256_load_si256  ((const __m256i*)&seeds[16]);
    y2 = _mm256_load_si256  ((const __m256i*)&seeds[24]);
    z1 = _mm256_load_si256  ((const __m256i*)&seeds[32]);
    z2 = _mm256_load_si256  ((const __m256i*)&seeds[40]);
    w1 = _mm256_load_si256  ((const __m256i*)&seeds[48]);
    w2 = _mm256_load_si256  ((const __m256i*)&seeds[54]);
#endif
    index = 15;
}



unsigned long XSRandom::cong(void)
{
    return (cx=69069*cx+362437);
}


#if __AVX512BW__
__m512i XSRandom::more()
#endif
#if ((!__AVX512BW__) & (__AVX2__))
std::pair<__m256i, __m256i> XSRandom::more()
#endif
{
#if __AVX512BW__
    __m512i t, u;

    t = _mm512_slli_epi32(x, 11);
    t = _mm512_xor_epi32(x, t);
    x = y;
    y = z;
    z = w;
    u = _mm512_srli_epi32(w, 19);
    w = _mm512_xor_epi32(w, u);
    u = _mm512_srli_epi32(t, 8);
    u = _mm512_xor_epi32(t, u);
    w = _mm512_xor_epi32(w, u);

    return w;
#endif
#if ((!__AVX512BW__) & (__AVX2__))
    __m256i t1, t2, u1, u2;

    t1 = _mm256_slli_epi32(x1, 11);
    t2 = _mm256_slli_epi32(x2, 11);
    t1 = _mm256_xor_si256 (x1, t1);
    t2 = _mm256_xor_si256 (x2, t2);
    x1 = y1;
    x2 = y2;
    y1 = z1;
    y2 = z2;
    z1 = w1;
    z2 = w2;
    u1 = _mm256_srli_epi32(w1, 19);
    u2 = _mm256_srli_epi32(w2, 19);
    w1 = _mm256_xor_si256 (w1, u1);
    w2 = _mm256_xor_si256 (w2, u2);
    u1 = _mm256_srli_epi32(t1, 8);
    u2 = _mm256_srli_epi32(t2, 8);
    u1 = _mm256_xor_si256 (t1, u1);
    u2 = _mm256_xor_si256 (t2, u2);
    w1 = _mm256_xor_si256 (w1, u1);
    w2 = _mm256_xor_si256 (w2, u2);

    return std::make_pair(w1, w2);
#endif
}



unsigned XSRandom::next()
{
#if __AVX512BW__
    __m512i r;

    index++;
    if (index == 16) {
        index = 0;
        r = more();
        _mm512_store_epi32(buffer, r);
    }
#endif
#if ((!__AVX512BW__) & (__AVX2__))
    // __m256i r1, r2;
    std::pair<__m256i, __m256i> r;

    index++;
    if (index == 16) {
        index = 0;
        r = more();
        _mm256_store_si256  ((__m256i*)&buffer[0], r.first);
        _mm256_store_si256  ((__m256i*)&buffer[8], r.second);
    }
#endif
    return buffer[index];
}


int XSRandom::fillArray(unsigned int *arr, int size)
{
#if __AVX512BW__
//#define PFD_XS 2
	if(size % 16 != 0)
	{
		printf("size is not a multiple of 16\n");
		return 0;
	}
	int i;
	__m512i r;

	for(i = 0; i < size; i+=16)
	{
//		_mm_prefetch((const char* ) &arr[i + PFD_XS * 16], _MM_HINT_T0);
		r = more();
		_mm512_store_epi32((__m256i*)(arr + i), more());
	}
#endif
#if ((!__AVX512BW__) & (__AVX2__))
    if(size % 16 != 0)
	{
		printf("size is not a multiple of 16\n");
		return 0;
	}
	int i;
	std::pair<__m256i, __m256i> r;
    std::pair<__m256i, __m256i> m;
    
	for(i = 0; i < size; i+=16)
	{
		r = more();
        m = more();
		_mm256_store_si256  ((__m256i*)(arr + i), m.first);
        _mm256_store_si256  ((__m256i*)(arr + i + 8), m.second);
	}
#endif
	return 1;
}