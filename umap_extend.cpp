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

Authors: Narendra Chaudhary <narendra.chaudhary@intel.com>; Sanchit Misra <sanchit.misra@intel.com>
*****************************************************************************************/



#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<fstream>
#include<iostream>
#include<omp.h>
#include<math.h>
#include <x86intrin.h>
#include "xsrandom.hpp"

namespace py = pybind11;

// def _optimize_layout_euclidean_single_epoch(
//     head_embedding,                          // <class 'numpy.ndarray'>, dtype = float32, shape = (990, 2)
//     tail_embedding,                          // <class 'numpy.ndarray'>, dtype = float32, shape = (990, 2)
//     head,                                    // <class 'numpy.ndarray'>, dtype = int, shape = (22116,)
//     tail,                                    // <class 'numpy.ndarray'>, dtype = int, shape = (22116,)
//     n_vertices,                              // <class 'int'>
//     epochs_per_sample,                       // <class 'numpy.ndarray'>, dtype = float64, shape = (22116,)
//     a,                                       // <class 'numpy.float64'>
//     b,                                       // <class 'numpy.float64'>
//     rng_state,                               // <class 'numpy.ndarray'>, dtype = int64, shape = (3,)
//     gamma,                                   // <class 'float'>
//     dim,                                     // <class 'int'>
//     move_other,                              // <class 'bool'>
//     alpha,                                   // <class 'float'>
//     epochs_per_negative_sample,              // <class 'numpy.ndarray'>, dtype = float64, shape = (22116,)
//     epoch_of_next_negative_sample,           // <class 'numpy.ndarray'>, dtype = float64, shape = (22116,)
//     epoch_of_next_sample,                    // <class 'numpy.ndarray'>, dtype = float64, shape = (22116,)
//     n,                                       // <class 'int'>
//     densmap_flag,                            // <class 'bool'>
//     dens_phi_sum,                            // <class 'numpy.ndarray'>, dtype = float32, shape = (1,)
//     dens_re_sum,                             // <class 'numpy.ndarray'>, dtype = float32, shape = (1,)
//     dens_re_cov,                             // <class 'int'>
//     dens_re_std,                             // <class 'int'>
//     dens_re_mean,                            // <class 'int'>
//     dens_lambda,                             // <class 'int'>
//     dens_R,                                  // <class 'numpy.ndarray'>, dtype = float32, shape = (1,)
//     dens_mu,                                 // <class 'numpy.ndarray'>, dtype = float32, shape = (1,)
//     dens_mu_tot,                             // <class 'int'>
// )

inline double clip(double val){
    if(val > 4.0)
        return 4.0;
    else if(val < -4.0)
        return -4.0;
    else
        return val;
}

// long tau_rand_int(long *state){
//     state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ ((((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19);
//     state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ ((((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25);
//     state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ ((((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11);

//     return int(state[0] ^ state[1] ^ state[2]);
// }

void cpp_optimize_layout_euclidean_single_epoch(py::array_t<float> head_embedding, py::array_t<float> tail_embedding, py::array_t<int> head, py::array_t<int> tail, \
    int n_vertices, py::array_t<double> epochs_per_sample, double a, double b, py::array_t<long> rng_state, float gamma, int dim, bool move_other, \
    float alpha, py::array_t<double> epochs_per_negative_sample, py::array_t<double> epoch_of_next_negative_sample, py::array_t<double> epoch_of_next_sample, \
    int n, bool densmap_flag, py::array_t<float> dens_phi_sum, py::array_t<float> dens_re_sum, int dens_re_cov, int dens_re_std, int dens_re_mean, int dens_lambda, \
    py::array_t<float> dens_R, py::array_t<float> dens_mu, int dens_mu_tot){
    
    py::buffer_info buf_head_embedding = head_embedding.request();
    py::buffer_info buf_tail_embedding = tail_embedding.request();
    py::buffer_info buf_head = head.request();
    py::buffer_info buf_tail = tail.request();

    py::buffer_info buf_epochs_per_sample = epochs_per_sample.request();
    // py::buffer_info buf_rng_state = rng_state.request();

    py::buffer_info buf_epochs_per_negative_sample = epochs_per_negative_sample.request();
    py::buffer_info buf_epoch_of_next_negative_sample = epoch_of_next_negative_sample.request();
    py::buffer_info buf_epoch_of_next_sample = epoch_of_next_sample.request();

    py::buffer_info buf_dens_phi_sum = dens_phi_sum.request();
    py::buffer_info buf_dens_re_sum = dens_re_sum.request();
    py::buffer_info buf_dens_R = dens_R.request();
    py::buffer_info buf_dens_mu = dens_mu.request();
    cout<<"Entered the CPP file !";
    float *head_embedding_ptr = (float *)buf_head_embedding.ptr;
    float *tail_embedding_ptr = (float *)buf_tail_embedding.ptr;
    int *head_ptr = (int *)buf_head.ptr;
    int *tail_ptr = (int *)buf_tail.ptr;

    double *epochs_per_sample_ptr = (double *)buf_epochs_per_sample.ptr;
    // long *rng_state_ptr = (long *)buf_rng_state.ptr;

    double *epochs_per_negative_sample_ptr = (double *)buf_epochs_per_negative_sample.ptr;
    double *epoch_of_next_negative_sample_ptr = (double *)buf_epoch_of_next_negative_sample.ptr;
    double *epoch_of_next_sample_ptr = (double *)buf_epoch_of_next_sample.ptr;

    float *dens_phi_sum_ptr = (float *)buf_dens_phi_sum.ptr;
    float *dens_re_sum_ptr = (float *)buf_dens_re_sum.ptr;
    float *dens_R_ptr = (float *)buf_dens_R.ptr;
    float *dens_mu_ptr = (float *)buf_dens_mu.ptr;

    // #pragma omp parallel for schedule(dynamic, 1000)
    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0; i < buf_epochs_per_sample.shape[0]; i++){              // 23029392
            if(epoch_of_next_sample_ptr[i] <= n){                           // 6928928
                int j = head_ptr[i];                                        // (Size = 4*6928928)
                int k = tail_ptr[i];                                        // (Size = 4*6928928)

                float *current, *other;
                current = &head_embedding_ptr[j*dim];                       // (Size = 16*990000)     
                other = &tail_embedding_ptr[k*dim];                         // (Size = 16*990000)

                // for(int d = 0; d < dim; d++){
                //     current[d] = head_embedding_ptr[j*dim + d];
                //     other[d] = tail_embedding_ptr[k*dim + d];
                // }

                float dist_squared=0.0f;
                for(int d = 0; d < dim; d++)
                    dist_squared += ((current[d] - other[d]) * (current[d] - other[d]));

                double grad_cor_coeff=0.0;
                if(densmap_flag){
                    double phi = 1.0 / (1.0 + a * pow(dist_squared, b));
                    double dphi_term = (a * b * pow(dist_squared, b - 1) / (1.0 + a * pow(dist_squared, b)));

                    double q_jk = phi / dens_phi_sum_ptr[k];
                    double q_kj = phi / dens_phi_sum_ptr[j];

                    double drk = q_jk * ((1.0 - b * (1 - phi)) / exp(dens_re_sum_ptr[k]) + dphi_term);
                    double drj = q_kj * ((1.0 - b * (1 - phi)) / exp(dens_re_sum_ptr[j]) + dphi_term);

                    double re_std_sq = dens_re_std * dens_re_std;

                    double weight_k = (dens_R_ptr[k] - dens_re_cov * (dens_re_sum_ptr[k] - dens_re_mean) / re_std_sq);
                    double weight_j = (dens_R_ptr[j] - dens_re_cov * (dens_re_sum_ptr[j] - dens_re_mean) / re_std_sq);

                    grad_cor_coeff = (
                        dens_lambda
                        * dens_mu_tot
                        * (weight_k * drk + weight_j * drj)
                        / (dens_mu_ptr[i] * dens_re_std)
                        / n_vertices
                    );
                }

                double grad_coeff=0.0;
                if(dist_squared > 0.0){
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0);
                    grad_coeff /= (a * pow(dist_squared, b) + 1.0);
                }
                else
                    grad_coeff = 0.0;

                double grad_d;
                for(int d = 0; d < dim; d++){
                    grad_d = clip(grad_coeff * (current[d] - other[d]));

                    if(densmap_flag)
                        grad_d += clip(2.0 * grad_cor_coeff * (current[d] - other[d]));

                    current[d] += (grad_d * alpha);
                    if(move_other)
                        other[d] += (-grad_d * alpha);
                }

                epoch_of_next_sample_ptr[i] += epochs_per_sample_ptr[i];                                // (Size = 2*8*6928928)

                int n_neg_samples = (int)((n - epoch_of_next_negative_sample_ptr[i]) / epochs_per_negative_sample_ptr[i]);       // (Size = 2*8*6928928)

                XSRandom prng;
                unsigned int list[(n_neg_samples/16 + 1)*16];
                prng.fillArray(list, (n_neg_samples/16 + 1)*16);
                for(int p = 0; p < n_neg_samples; p++){                                         // 5 times on average

                    // k = tau_rand_int(rng_state_ptr) % n_vertices;                               // ((n % M) + M) % M
                    // k = ((tau_rand_int(rng_state_ptr) % n_vertices) + n_vertices) % n_vertices;                               // ((n % M) + M) % M
                    k = list[p] % n_vertices;

                    other = &tail_embedding_ptr[k*dim];

                    dist_squared=0.0f;
                    for(int d = 0; d < dim; d++)
                        dist_squared += ((current[d] - other[d]) * (current[d] - other[d]));                  

                    if(dist_squared > 0.0){
                        grad_coeff = 2.0 * gamma * b;
                        grad_coeff /= ((0.001 + dist_squared) * (a * pow(dist_squared, b) + 1.0));
                    }
                    else if(j == k)
                        continue;
                    else
                        grad_coeff = 0.0;

                    for(int d = 0; d < dim; d++){
                        if(grad_coeff > 0.0)
                            grad_d = clip(grad_coeff * (current[d] - other[d]));
                        else
                            grad_d = 4.0;
                        current[d] += (grad_d * alpha);
                    }
                }                 

                epoch_of_next_negative_sample_ptr[i] += (n_neg_samples * epochs_per_negative_sample_ptr[i]);
            }
        }
    }
}


PYBIND11_MODULE(umap_extend, handle){
    handle.doc() = "This is the module for _optimize_layout_euclidean_single_epoch() function.";
    handle.def("cpp_optimize_layout_euclidean_single_epoch", &cpp_optimize_layout_euclidean_single_epoch);
}