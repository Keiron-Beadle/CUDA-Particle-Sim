/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>
#include "curand.h"
#include "curand_kernel.h"
#include"hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "particle.h"

typedef unsigned int uint;
typedef unsigned char uchar;
dim3 grid_size = dim3(8, 8);
dim3 block_size = dim3(128, 8);
__device__ __constant__ const float PARTICLE_SIZE = 0.01f;
__device__ __constant__ const int WALL_COUNT = 6;
__device__ __constant__ const int NUM_OF_PARTICLES = 29000;
__device__ __constant__ const int NUM_OF_DRAWN_PARTICLES = 80 + WALL_COUNT;
__device__ __constant__ const int NUM_OF_SPHERES = WALL_COUNT + NUM_OF_PARTICLES;
__device__ __constant__ const int NUM_OF_COLLISION_THREADS = NUM_OF_PARTICLES / 500;

int* d_frame_collisions;

__device__ hitable** d_world;
__device__ hitable** d_list;

cudaArray* d_imageArray = 0;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ ) 
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting 
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 castRay(const ray& r, hitable** world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        vec3 material = vec3(rec.color.x(), rec.color.y(), rec.color.z());
        vec3 normal = vec3(rec.normal.x(), rec.normal.y(), rec.normal.z());
        return 0.5f * (material * 1.0f + normal * 0.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

__global__ void create_world(hitable** d_list, hitable** d_world, curandState* randomState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
	//This code is generated a "box" or a "room". It is given the appearance of a cube by using 5 humongous spheres
        *(d_list) = new sphere(vec3(0, -10000, 0), 10000, vec3(0.639, 0.639, 0.639));
        *(d_list + 1) = new sphere(vec3(0, 10001.5, 0), 10000, vec3(0.639, 0.639, 0.639));
        *(d_list + 2) = new sphere(vec3(-10000.5, 0, 0), 10000, vec3(0.431, 0.431, 0.431));
        *(d_list + 3) = new sphere(vec3(10000.5, 0, 0), 10000, vec3(0.431, 0.431, 0.431));
        *(d_list + 4) = new sphere(vec3(0, 0, -10000.5), 10000, vec3(0.250, 0.250, 0.250));
	//This code is spawning a model at the top of the "shower" that will dispense particles
        *(d_list + 5) = new sphere(vec3(0, 1.55f, 0), 0.1, vec3(0.258, 0.611, 1));
        int o = 6;
	//This code generates particles in a random area within the shower head.
        for (int i = 0; i < NUM_OF_PARTICLES; i++) {
            curand_init(2001, i, 0, &randomState[i]);
            float rand_x = curand_uniform(&randomState[i]) * 0.1f - 0.05f;
            float rand_z = curand_uniform(&randomState[i]) * 0.1f - 0.05f;
            *(d_list + i + o) = new particle(vec3(rand_x, 1.5f, rand_z), PARTICLE_SIZE, vec3(1, 0, 0), i, randomState);
        }
        *d_world = new hitable_list(d_list, NUM_OF_DRAWN_PARTICLES);
    }
}

__global__ void d_update_collision(hitable** d_list, int* d_frame_collisions) {    
    uint width = (NUM_OF_SPHERES / NUM_OF_COLLISION_THREADS);
    uint idx = threadIdx.x;  
    for (int i = 0; i < width; ++i) {
        int current = i * NUM_OF_COLLISION_THREADS + idx;
        if (current <= 5 || current >= NUM_OF_SPHERES) { continue; }
        particle* p = (particle*)*(d_list + current);
        vec3 pos = p->center;
        if (pos.y() < 0.0f) {
            p->reset();
            atomicAdd(d_frame_collisions, 1);
            for (int j = 0; j < width; j++) {
                int j_index = j * NUM_OF_COLLISION_THREADS + idx;
                if (j_index <= 5 || j_index >= NUM_OF_SPHERES) { continue; }
                particle* pj = (particle*)*(d_list + j_index);
                if (pj->collider == p) {
                    pj->reset();
                    pj->enable();
                }
            }
        }

        //PARTICLE PARTICLE COLLISION
        for (int j = i; j < width; j++) {
            int j_index = j * NUM_OF_COLLISION_THREADS + idx;
            if (j_index <= 5 || j_index >= NUM_OF_SPHERES || j == i) { continue; }
            particle* pj = (particle*)*(d_list + j_index);
            bool collided = p->collide(pj, j_index, 0.002f);
            if (collided) {
                pj->disable();
                for (int k = 0; k < width; k++) {
                    int k_index = k * NUM_OF_COLLISION_THREADS + idx;
                    if (k_index <= 5 || k_index >= NUM_OF_SPHERES) { continue; }
                    particle* pk = (particle*)*(d_list + k_index);
                    if (pk->collider == pj) {
                        pk->collider = p;
                    }
                }
                
            }
        }
    }
    __syncthreads();
}

__device__ void d_update_movement(hitable** d_list, uint index) {
    particle* p = (particle*)*(d_list + index);
    p->movement_tick();
}

__device__ void d_update_temperature(hitable** d_list, uint index) {
    particle* p = (particle*)*(d_list + index);
    p->temperature_tick();
}

__global__ void d_particles_update(hitable** d_list) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * blockDim.x + x + 6;
    if (i > NUM_OF_SPHERES) { return; }
    d_update_movement(d_list, i);
    d_update_temperature(d_list, i);
}

__global__ void d_switch_mode(hitable** d_list) {
    for (int i = 6; i < NUM_OF_SPHERES; i++) {
        particle* p = (particle*)*(d_list + i);
        p->switch_mode();
    }
}

__global__ void d_render(uchar4* d_output, hitable** d_world, uint width, uint height) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;
    if (!((x < width) && (y < height))) { return; }
    float u = x / (float)width;
    float v = y / (float)height;
    u = 2.0 * u - 1.0;
    v = -(2.0 * v - 1.0);
    u *= width / (float)height;

    u *= 2.0;
    v *= 2.0;

    vec3 eye = vec3(0, 1.0, 0.5);
    float distFrEye2Img = 0.5;
    vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
    ray r;
    r.O = eye;
    r.Dir = pixelPos - eye;
    vec3 col = castRay(r, d_world);
    float red = col.x();
    float green = col.y();
    float blue = col.z();
    d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
}

__global__ void free_world(hitable** d_list, hitable** d_world) {
    for (int i = 0; i < NUM_OF_SPHERES; i++) {
        delete* (d_list + i);
    }
    delete* d_world;
}

extern "C" void freeTexture() {
    checkCudaErrors(cudaDeviceSynchronize()); //sync to get d_frame_collisions finished
    std::cout << "TOTAL FLOOR COUNT : " << d_frame_collisions[0] << '\n';
    cudaFree(d_frame_collisions);
    checkCudaErrors(cudaFreeArray(d_imageArray));
}

extern "C" void initWorld() {
    curandState* state;
    checkCudaErrors(cudaMalloc(&state, NUM_OF_PARTICLES * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&d_frame_collisions, sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&state, sizeof(curandState)));
    checkCudaErrors(cudaMallocManaged((void**)&d_list, NUM_OF_SPHERES * sizeof(hitable*)));
    checkCudaErrors(cudaMallocManaged((void**)&d_world, sizeof(hitable*)));
    create_world << <1, 1 >> > (d_list, d_world, state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(state);
}

//Switch mode is toggled when switching between visualisaiton modes
extern "C" void switch_mode() {
    d_switch_mode << <1, 1 >> > (d_list);
}

// render image using CUDA
extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize,
    uchar4 * output) {
    //Timing code is commented out for demo purposes to increase performance.
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    d_particles_update << <grid_size, block_size >> > (d_list);
    //cudaEventRecord(start, 0);
    d_update_collision << <1, NUM_OF_COLLISION_THREADS >> > (d_list, d_frame_collisions);
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //float elapsedTime;
    //cudaEventElapsedTime(&elapsedTime, start, stop);
    //std::cout << elapsedTime << '\n';
    d_render << <gridSize, blockSize >> > (output, d_world, width, height);
    getLastCudaError("kernel failed");
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
}

#endif