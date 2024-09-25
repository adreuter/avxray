#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>

#include "vec.h"

#define ARRSZ(arr) (sizeof((arr)) / sizeof((arr)[0]))

static inline vf3
color(vf3 dir, uint32_t ssz, const float scs[ssz / VFSZ][3][VFSZ], const float srs[ssz / VFSZ][VFSZ])
{
	const vf maxdist = vffrom(INFINITY);
	vf dist = maxdist;
	vf3 sc_hit = vf3from(vffrom(INFINITY), vffrom(INFINITY), vffrom(INFINITY));
	uint32_t sl, si;
	
	for (sl = 0; sl < ssz / VFSZ; sl++) {	
		for (si = 0; si < VFSZ; si++) {
			/* TODO: probably better to completely fetch 
			 * one lane at a time and then in-pl shuffle around */
			const vf3 sc = vf3from(vffrom(scs[sl][0][si]),
					       vffrom(scs[sl][1][si]),
					       vffrom(scs[sl][2][si]));
			const float sr = srs[sl][si];
			
			/* calculate ray-sphere intersection */
			const vf a = vf3dot(dir, dir);
			const vf h = vf3dot(dir, sc);
			const vf c = vfsub(vf3dot(sc, sc), vffrom(sr * sr));
			const vf d = vfsub(vfmul(h, h), vfmul(a, c));
				
			const vf ndist = vfdiv(vfsub(h, vfsqrt(d)), a);
			/* ndist >= 0.0 && ndist < dist */
			const vf iscloser = vfand(vfgeq(ndist, vffrom(0.0)), vfgeq(dist, ndist));
			
			dist     = _mm256_blendv_ps(dist, ndist, iscloser);
			sc_hit.x = _mm256_blendv_ps(sc_hit.x, sc.x, iscloser);
			sc_hit.y = _mm256_blendv_ps(sc_hit.y, sc.y, iscloser);
			sc_hit.z = _mm256_blendv_ps(sc_hit.z, sc.z, iscloser);
		}
	}
	
	/* constants */	
	const vf3 sundir = vf3from(vffrom(0.0), vffrom(1.0), vffrom(0.0));
       	const vf3 color = vf3from(vffrom(0.0), vffrom(1.0), vffrom(1.0));
	/* improved lambertian diffuse shading */
	/* divide by sphere radius instead of vf3normal(..)!!*/
	const vf3 snl = vf3normal(vf3sub(vf3mul(dir, dist), sc_hit));
	const vf intensity = _mm256_max_ps(vffrom(0.0), vf3dot(snl, sundir));
	/* could be repeated more times for even more exaggerated shading */
	const vf diff = intensity + (1-intensity) * intensity;
	return vf3mul(color, diff);
}

/* camera center implicitly at (0, 0, 0) */
static inline void
render(const uint32_t w, const uint32_t h, const float vh, const float fl,
       float bf[h][w / VFSZ][3][VFSZ],
       uint32_t ssz, const float scs[ssz / VFSZ][3][VFSZ], const float srs[ssz / VFSZ][VFSZ])
{
	/* camera */
	const float vw = vh * (((float) w) / h);
	const vf3 vu = vf3from(vffrom(vw), vffrom(0.0), vffrom(0.0));
	const vf3 vv = vf3from(vffrom(0.0), vffrom(-vh), vffrom(0.0));
	const vf3 du = vf3div(vu, vffrom(w));
	const vf3 dv = vf3div(vv, vffrom(h));
	const vf3 ul = vf3sub(vf3from(vffrom(0.0), vffrom(0.0), vffrom(-fl)),
			      vf3mul(vf3add(vu, vv), vffrom(0.5)));
	const vf3 p00 = vf3add(ul, vf3mul(vf3add(du, dv), vffrom(0.5)));
	/* TODO: set x from 0 to LANESZ - 1*/
	const float xarr[VFSZ] = {0, 1, 2, 3, 4, 5, 6, 7};
	const vf x = vfload(xarr); /*  a vi type would be better here */
	/* indices */
	uint32_t xi, y;
	
	for (y = 0; y < h; y++) {
		for (xi = 0; xi < w / VFSZ; xi++) {
			/* camera ray direction */
			const vf3 dir = vf3add(p00, 
                            		vf3add(vf3mul(du, vfadd(x, vffrom(xi * VFSZ))),
                            		       vf3mul(dv, vffrom(y))));
			/* determine color */
			vf3 c = color(dir, ssz, scs, srs);
			/* write color */
            		vfstore(bf[y][xi][0], vfsqrt(c.x));
            		vfstore(bf[y][xi][1], vfsqrt(c.y));
           		vfstore(bf[y][xi][2], vfsqrt(c.z));
		}
	}
}

int
main(void)
{
	const uint32_t w = 400;
	const uint32_t h = 225;
	const float scs[][3][VFSZ] = {{{ 0.0,  0.0,  -1.0,  1.0, 0.0, 0.0, 0.0, 0.0},
				       { 0.0, -100.5, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0}, 
				       {-1.2, -1.0,  -1.0, -1.0, 0.0, 0.0, 0.0, 0.0}}};
	const float srs[][VFSZ]    = {{  0.5,  100.0, 0.5,  0.5, NAN, NAN, NAN, NAN}};

	float (*bf)[h][w / VFSZ][3][VFSZ];
	
	/* w must be divisible by VFSZ */
	assert(w % VFSZ == 0);
	bf = malloc(sizeof(*bf));
	if (!bf)
		perror("failed to allocate image buffer");
	
	render(w, h, 2.0, 1.0, *bf, ARRSZ(scs)*VFSZ, scs, srs);
	
	/* convert bf to ppm and print to stdout */	
    	uint32_t x, y, i;
	printf("P3\n%u %u\n255\n", w, h);
	for (y = 0; y < h; y++)
		for (x = 0; x < w / VFSZ; x++)
			for (i = 0; i < VFSZ; i++)
				printf("%u %u %u\n",
			       	       (uint8_t) (255.999 * (*bf)[y][x][0][i]),
			       	       (uint8_t) (255.999 * (*bf)[y][x][1][i]),
			       	       (uint8_t) (255.999 * (*bf)[y][x][2][i]));

	free(bf);
	return 0;
}

