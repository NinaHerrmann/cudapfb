#pragma once
/*
 * rng.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#pragma once

#ifdef __CUDACC__
#include <thrust/random.h>
#endif
#include <random>
#ifdef __CUDACC__
 /**
  * \brief Macro for function type qualifiers __host__ __device__.
  *
  * Macro for function type qualifiers __host__ __device__. This macro is only
  * define when compiled with the Nvidia C compiler nvcc because ordinary C/C++
  * compiler will complain about function type qualifiers.
  */
#define MSL_USERFUNC __host__ __device__
#else
#define MSL_USERFUNC
#endif


namespace msl {

	/**
	 * \brief Class Rng represents a pseudo random number generator that can be called by both
	 *        the CPU and the GPU. Uses std::default_random_engine and
	 *        std::uniform_real_distribution for the CPU side, and thrust::default_random_engine
	 *        and thrust::uniform_real_distribution on the GPU side.
	 */
	class Rng
	{
	public:
		/**
		 * \brief Default constructor.
		 */
		MSL_USERFUNC
			Rng()
			: min(0.f), max(1.f), rng(hash(1)), dist(0.f, 1.f)
		{
		}

		/**
		 * \brief Creates a pseudo random number generator with minimum value \em minVal
		 *        and maximum value \em maxVal.
		 *
		 * @param minVal The minimum value.
		 * @param maxVal The maximum value.
		 */
		// TODO replaced msl::getUniqueID() with 1
		MSL_USERFUNC
			Rng(float minVal, float maxVal)
			: min(minVal), max(maxVal), rng(hash(1)), dist(minVal, maxVal)
		{
		}

		/**
		 * \brief Creates a pseudo random number generator with minimum value \em minVal
		 *        and maximum value \em maxVal and sets a new seed \em seed.
		 *
		 * @param minVal The minimum value.
		 * @param maxVal The maximum value.
		 * @param seed The new seed.
		 */
		MSL_USERFUNC
			Rng(float minVal, float maxVal, size_t seed)
			: min(minVal), max(maxVal), rng(seed), dist(minVal, maxVal)
		{
		}

		/**
		 * Returns the next pseudo random number.
		 *
		 * @return The next pseudo random number.
		 */
		MSL_USERFUNC
			float operator() ()
		{
			return dist(rng);
		}

	private:
		float min, max;
#ifdef __CUDA_ARCH__
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist;
#else
		std::default_random_engine rng;
		std::uniform_real_distribution<float> dist;
#endif

		MSL_USERFUNC
			size_t hash(size_t a) const
		{
			a = (a + 0x7ed55d16) + (a << 12);
			a = (a ^ 0xc761c23c) ^ (a >> 19);
			a = (a + 0x165667b1) + (a << 5);
			a = (a + 0xd3a2646c) ^ (a << 9);
			a = (a + 0xfd7046c5) + (a << 3);
			a = (a ^ 0xb55a4f09) ^ (a >> 16);
			return a;
		}
	};

} // namespace msl


