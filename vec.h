/* requires: immintrin.h */
#define VFSZ (sizeof(vf) / sizeof(float))

typedef __m256 vf;

typedef struct {
	vf x, y, z;
} vf3;

static inline vf
vffrom(const float f) { return _mm256_set1_ps(f); }

static inline vf
vfload(const float *f) { return _mm256_load_ps(f); }

static inline void
vfstore(float *dst, const vf s) { _mm256_store_ps(dst, s); }

static inline vf
vfadd(const vf s1, const vf s2) { return _mm256_add_ps(s1, s2); }

static inline vf
vfsub(const vf s1, const vf s2) { return _mm256_sub_ps(s1, s2); }

static inline vf
vfmul(const vf s1, const vf s2) { return _mm256_mul_ps(s1, s2); }

static inline vf
vfdiv(const vf s1, const vf s2) { return _mm256_div_ps(s1, s2); }

static inline vf
vfand(const vf s1, const vf s2) { return _mm256_and_ps(s1, s2); }

static inline vf
vfsqrt(const vf s) { return _mm256_sqrt_ps(s); }

static inline vf
vfrsqrt(const vf s) { return _mm256_rsqrt_ps(s); }

static inline vf
vfgeq(const vf s1, const vf s2) { return _mm256_cmp_ps(s1, s2, _CMP_GE_OS); }

static inline vf3
vf3from(const vf x, const vf y, const vf z)
{
	vf3 res = {.x = x, .y = y, .z = z};
	return res;
}

static inline vf3
vf3add(const vf3 s1, const vf3 s2)
{
	vf3 res = {
        	.x = vfadd(s1.x, s2.x),
        	.y = vfadd(s1.y, s2.y),
        	.z = vfadd(s1.z, s2.z)
    	};
	return res;
}

static inline vf3
vf3sub(const vf3 s1, const vf3 s2)
{
	vf3 res = {
        	.x = vfsub(s1.x, s2.x),
        	.y = vfsub(s1.y, s2.y),
        	.z = vfsub(s1.z, s2.z)
    	};
	return res;
}

static inline vf3
vf3mul(const vf3 s1, const vf c)
{
	vf3 res = {
        	.x = vfmul(s1.x, c),
        	.y = vfmul(s1.y, c),
        	.z = vfmul(s1.z, c)
    	};
	return res;
}

static inline vf3
vf3div(const vf3 s1, const vf c)
{
	vf3 res = {
        	.x = vfdiv(s1.x, c),
        	.y = vfdiv(s1.y, c),
        	.z = vfdiv(s1.z, c)
    	};
	return res;
}

static inline vf3
vf3and(const vf3 s1, const vf c)
{
	vf3 res = {
        	.x = vfand(s1.x, c),
        	.y = vfand(s1.y, c),
        	.z = vfand(s1.z, c)
    	};
	return res;
}

static inline vf
vf3dot(const vf3 s1, const vf3 s2)
{
	return vfadd(vfadd(vfmul(s1.x, s2.x), 
			   vfmul(s1.y, s2.y)), 
		     vfmul(s1.z, s2.z));
}

static inline vf3
vf3normal(const vf3 s)
{
	return vf3div(s, vfsqrt(vf3dot(s, s)));
}

static inline void
vfprint(const vf vf)
{
	float f[VFSZ];
	vfstore(f, vf);

	printf("%f %f %f %f %f %f %f %f\n",
	       f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
}
