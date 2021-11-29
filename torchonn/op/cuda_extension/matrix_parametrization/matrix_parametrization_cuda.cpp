/*
 * @Author: Jiaqi Gu
 * @Date: 2020-06-04 14:16:01
 * @LastEditors: Jiaqi Gu (jqgu@utexas.edu)
 * @LastEditTime: 2021-11-29 03:01:34
 */
#include <torch/torch.h>
#include<bits/stdc++.h>
using namespace std;
using namespace torch::indexing;

#define PI (3.141592653589793)
#define abs(x) (x >= 0 ? x : -x)
#define ERROR (1e-6)

template <typename T>
void unitaryRotationCUDALauncher(const int N, const int num_block, const int num_thread, T* row_p, T* row_q, const T* cos_phi, const T* sin_phi);

template <typename T>
void unitaryReconstructFrancisCUDALauncher(const int BS, const int N, T *U, const T* cos_phi_mat, const T* sin_phi_mat);

template <typename T>
void unitaryReconstructClementsCUDALauncher(const int BS, const int N, T *U, const T* cos_phi_mat, const T* sin_phi_mat, const T* delta_list_0, const T* delta_list_N);

template <typename T>
void unitaryReconstructReckCUDALauncher(const int BS, const int N, T *U, const T* cos_phi_mat, const T* sin_phi_mat);

template <typename T>
void unitaryDecomposeFrancisCUDALauncher(const int BS, const int N, T *U, T* delta_list, T* phi_mat);

template <typename T>
void unitaryDecomposeClementsCUDALauncher(const int BS, const int N, T *U, T* delta_list, T* phi_mat);

template <typename T>
void unitaryDecomposeReckCUDALauncher(const int BS, const int N, T *U, T* delta_list, T* phi_mat);

template <typename T>
inline T calPhi(const T u1, const T u2)
{
    T u1_abs = abs(u1);
    T u2_abs = abs(u2);
    T phi = 0;
    int cond = ((u1_abs >= ERROR) << 1) | (u2_abs >= ERROR);
    switch(cond)
    {
        case 0: phi = 0; break;
        case 1: phi = u2 > ERROR ? -0.5 * PI : 0.5 * PI; break;
        case 2: phi = u1 > ERROR ? 0 : -PI; break;
        case 3: phi = std::atan2(-u2, u1); break;
        default: break;
    }

    // if(u1_abs < ERROR && u2_abs < ERROR)
    //     phi = 0;
    // else if (u1_abs >= ERROR && u2_abs < ERROR)
    //     phi = u1 > ERROR ? 0 : -PI;
    // else if (u1_abs < ERROR && u2_abs >= ERROR)
    //     phi = u2 > ERROR ? -0.5 * PI : 0.5 * PI;
    // else
    //     phi = std::atan2(-u2, u1);
    return phi;
}

void decompose_francis_forward(
    at::Tensor U,
    at::Tensor delta_list,
    at::Tensor phi_mat
    )
{
    // TORCH_CHECK(U.type().is_cuda(), "U must be a CUDA tensor")
    TORCH_CHECK(U.size(-2) == U.size(-1), "U must be a square tensor")
    TORCH_CHECK(U.is_contiguous(), "U must be contiguous()")
    TORCH_CHECK(delta_list.size(-1) == U.size(-1), "delta_list should have length N")
    TORCH_CHECK(phi_mat.size(-2) == phi_mat.size(-1) && phi_mat.size(-1) == U.size(-1), "phi mat should have size NxN")


    auto N = U.size(-1);
    U = U.contiguous();
    phi_mat = phi_mat.contiguous(); // left-upper triangular matrix
    delta_list = delta_list.contiguous();
    int BS = U.numel() / (N * N);
    AT_DISPATCH_FLOATING_TYPES(U.scalar_type(), "unitaryDecomposeFrancisCUDALauncher", ([&] {
        unitaryDecomposeFrancisCUDALauncher<scalar_t>(
            BS,
            N,
            U.data_ptr<scalar_t>(),
            delta_list.data_ptr<scalar_t>(),
            phi_mat.data_ptr<scalar_t>());
    }));

    #if 0
    for (int i = 0; i < N - 1; ++i)
    {
        //decomposeKernel
        for(int j = i; j < N - 1; ++j)
        {
            int p = i;
            int q = N - 1 - j + i;
            double u1 = U.index({p, p}).item().toDouble();
            double u2 = U.index({p, q}).item().toDouble();
            double phi = calPhi<double>(u1, u2);
            phi_mat.index({i, j-i}) = phi;
            // phi in [-pi, pi]
            double c = std::cos(phi);
            double s = phi <= 0 ? -std::sqrt(1 - c*c) : std::sqrt(1 - c*c);
            // double s = std::sin(phi);
            auto col_p = U.index({Slice(i,None), p});
            auto col_q = U.index({Slice(i,None), q});
            auto col_p_cos = col_p * c;
            auto col_p_sin = col_p * s;
            auto col_q_cos = col_q * c;
            auto col_q_sin = col_q * s;

            auto temp = col_p_cos - col_q_sin;
            U.index({Slice(i,None), q}) = col_p_sin + col_q_cos;
            U.index({Slice(i,None), p}) = temp;
        }
        delta_list.index({i}) = U.index({i, i});
    }
    delta_list.index({N-1}) = U.index({N-1, N-1});
    #endif
}

torch::Tensor reconstruct_francis_forward(
//   torch::Tensor U,
  torch::Tensor delta_list,
  torch::Tensor phi_mat
)
{
    // TORCH_CHECK(U.type().is_cuda(), "U must be a CUDA tensor")
    // TORCH_CHECK(U.size(-2) == U.size(-1), "U must be a square tensor")
    // TORCH_CHECK(U.is_contiguous(), "U must be contiguous")
    TORCH_CHECK(phi_mat.is_contiguous(), "phi mat must be contiguous")
    // TORCH_CHECK(delta_list.size(-1) == U.size(-1), "delta_list should have length N")
    // TORCH_CHECK(phi_mat.size(-2) == phi_mat.size(-1) && phi_mat.size(-1) == U.size(-1), "phi mat should have size NxN")

    int N = delta_list.size(-1);
    auto phi_mat_cos = torch::cos(phi_mat); // left-upper triangular matrix
    auto phi_mat_sin = torch::sin(phi_mat);
    auto W = torch::ones_like(delta_list).diag_embed().contiguous();
    int BS = delta_list.numel() / N;
    phi_mat = phi_mat.contiguous();
    AT_DISPATCH_FLOATING_TYPES(W.scalar_type(), "unitaryReconstructFrancisCUDALauncher", ([&] {
        unitaryReconstructFrancisCUDALauncher<scalar_t>(
            BS,
            N,
            W.data_ptr<scalar_t>(),
            phi_mat_cos.data_ptr<scalar_t>(),
            phi_mat_sin.data_ptr<scalar_t>());
    }));

    // unitaryReconstructCUDALauncher(
    //     N,
    //     U.data<double>(),
    //     phi_mat_cos.data<double>(),
    //     phi_mat_sin.data<double>()
    // );
    #if 0
    int num_thread = 512;
    int num_block = (2 * N + num_thread - 1) / num_thread;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N - i - 1; ++j)
        {
            // double c = phi_mat_cos.index({i, j}).item().toDouble();
            // double s = phi_mat_sin.index({i, j}).item().toDouble();
            auto c = phi_mat_cos.index({i, j});
            auto s = phi_mat_sin.index({i, j});
            int p = i;
            int q = N - j - 1;
            // auto row_p = U.index({p, Slice()});
            // auto row_q = U.index({q, Slice()});
            // auto row_p_cos = row_p * c;
            // auto row_p_sin = row_p * s;
            // auto row_q_cos = row_q * c;
            // auto row_q_sin = row_q * s;
            // U.index({p, Slice()}) = row_p_cos - row_q_sin;
            // U.index({q, Slice()}) = row_p_sin + row_q_cos;
            // AT_DISPATCH_FLOATING_TYPES(U.type(), "unitaryRotationCUDALauncher", ([&] {
            //     unitaryRotationCUDALauncher<scalar_t>(
            //         N,
            //         U.index({p, Slice()}).data<scalar_t>(),
            //         U.index({q, Slice()}).data<scalar_t>(),
            //         c.data<scalar_t>(),
            //         s.data<scalar_t>());
            // }));
            unitaryRotationCUDALauncher(
                N,
                num_block,
                num_thread,
                U.index({p, Slice()}).contiguous().data<double>(),
                U.index({q, Slice()}).contiguous().data<double>(),
                c.data<double>(),
                s.data<double>()
            );
        }
    }
    #endif
    return W.mul_(delta_list.unsqueeze(-1));


}

void decompose_clements_forward(
  at::Tensor U,
  at::Tensor delta_list,
  at::Tensor phi_mat
)
{
    // TORCH_CHECK(U.type().is_cuda(), "U must be a CUDA tensor")
    TORCH_CHECK(U.size(-2) == U.size(-1), "U must be a square tensor")
    TORCH_CHECK(U.is_contiguous(), "U must be contiguous()")
    TORCH_CHECK(delta_list.size(-1) == U.size(-1), "delta_list should have length N")
    TORCH_CHECK(phi_mat.size(-2) == phi_mat.size(-1) && phi_mat.size(-1) == U.size(-1), "phi mat should have size NxN")


    auto N = U.size(-1);
    U = U.contiguous();
    phi_mat = phi_mat.contiguous(); // clements-style interleaved checkerboard
    delta_list = delta_list.contiguous();
    int BS = U.numel() / (N * N);
    AT_DISPATCH_FLOATING_TYPES(U.scalar_type(), "unitaryDecomposeClementsCUDALauncher", ([&] {
        unitaryDecomposeClementsCUDALauncher<scalar_t>(
            BS,
            N,
            U.data_ptr<scalar_t>(),
            delta_list.data_ptr<scalar_t>(),
            phi_mat.data_ptr<scalar_t>());
    }));

    #if 0
    for (int i = 0; i < N - 1; ++i)
    {
        //decomposeKernel
        for(int j = i; j < N - 1; ++j)
        {
            int p = i;
            int q = N - 1 - j + i;
            double u1 = U.index({p, p}).item().toDouble();
            double u2 = U.index({p, q}).item().toDouble();
            double phi = calPhi<double>(u1, u2);
            phi_mat.index({i, j-i}) = phi;
            // phi in [-pi, pi]
            double c = std::cos(phi);
            double s = phi <= 0 ? -std::sqrt(1 - c*c) : std::sqrt(1 - c*c);
            // double s = std::sin(phi);
            auto col_p = U.index({Slice(i,None), p});
            auto col_q = U.index({Slice(i,None), q});
            auto col_p_cos = col_p * c;
            auto col_p_sin = col_p * s;
            auto col_q_cos = col_q * c;
            auto col_q_sin = col_q * s;

            auto temp = col_p_cos - col_q_sin;
            U.index({Slice(i,None), q}) = col_p_sin + col_q_cos;
            U.index({Slice(i,None), p}) = temp;
        }
        delta_list.index({i}) = U.index({i, i});
    }
    delta_list.index({N-1}) = U.index({N-1, N-1});
    #endif
}

torch::Tensor reconstruct_clements_forward(
//   torch::Tensor U,
  torch::Tensor delta_list,
  torch::Tensor phi_mat
)
{
    // TORCH_CHECK(U.type().is_cuda(), "U must be a CUDA tensor")
    // TORCH_CHECK(U.size(-2) == U.size(-1), "U must be a square tensor")
    // TORCH_CHECK(U.is_contiguous(), "U must be contiguous")
    TORCH_CHECK(phi_mat.is_contiguous(), "phi mat must be contiguous")
    // TORCH_CHECK(delta_list.size(-1) == U.size(-1), "delta_list should have length N")
    // TORCH_CHECK(phi_mat.size(-2) == phi_mat.size(-1) && phi_mat.size(-1) == U.size(-1), "phi mat should have size NxN")

    int N = delta_list.size(-1);
    auto phi_mat_cos = torch::cos(phi_mat); // clement-style interleaved checkerboard
    auto phi_mat_sin = torch::sin(phi_mat);
    auto W = torch::ones_like(delta_list).diag_embed().contiguous();
    int BS = delta_list.numel() / N;
    phi_mat = phi_mat.contiguous();
    // handle the first row negation here
    // W.index({"...", N-1, N-1}).mul_(delta_list.index({"...", N-1}));
    AT_DISPATCH_FLOATING_TYPES(W.scalar_type(), "unitaryReconstructClementsCUDALauncher", ([&] {
        unitaryReconstructClementsCUDALauncher<scalar_t>(
            BS,
            N,
            W.data_ptr<scalar_t>(),
            phi_mat_cos.data_ptr<scalar_t>(),
            phi_mat_sin.data_ptr<scalar_t>(),
            delta_list.index({"...", 0}).contiguous().data_ptr<scalar_t>(),
            delta_list.index({"...", N-1}).contiguous().data_ptr<scalar_t>()
            // delta_list.data_ptr<scalar_t>()
            );
    }));

    // U.copy_(W);
    return W;

}


void decompose_reck_forward(
  at::Tensor U,
  at::Tensor delta_list,
  at::Tensor phi_mat
)
{
    // TORCH_CHECK(U.type().is_cuda(), "U must be a CUDA tensor")
    TORCH_CHECK(U.size(-2) == U.size(-1), "U must be a square tensor")
    TORCH_CHECK(U.is_contiguous(), "U must be contiguous()")
    TORCH_CHECK(delta_list.size(-1) == U.size(-1), "delta_list should have length N")
    TORCH_CHECK(phi_mat.size(-2) == phi_mat.size(-1) && phi_mat.size(-1) == U.size(-1), "phi mat should have size NxN")


    auto N = U.size(-1);
    U = U.contiguous();
    phi_mat = phi_mat.contiguous(); // reck-style upper triangular array
    delta_list = delta_list.contiguous();
    int BS = U.numel() / (N * N);
    AT_DISPATCH_FLOATING_TYPES(U.scalar_type(), "unitaryDecomposeReckCUDALauncher", ([&] {
        unitaryDecomposeReckCUDALauncher<scalar_t>(
            BS,
            N,
            U.data_ptr<scalar_t>(),
            delta_list.data_ptr<scalar_t>(),
            phi_mat.data_ptr<scalar_t>());
    }));

}


torch::Tensor reconstruct_reck_forward(
//   torch::Tensor U,
  torch::Tensor delta_list,
  torch::Tensor phi_mat
)
{
    // TORCH_CHECK(U.type().is_cuda(), "U must be a CUDA tensor")
    // TORCH_CHECK(U.size(-2) == U.size(-1), "U must be a square tensor")
    // TORCH_CHECK(U.is_contiguous(), "U must be contiguous")
    TORCH_CHECK(phi_mat.is_contiguous(), "phi mat must be contiguous")
    // TORCH_CHECK(delta_list.size(-1) == U.size(-1), "delta_list should have length N")
    // TORCH_CHECK(phi_mat.size(-2) == phi_mat.size(-1) && phi_mat.size(-1) == U.size(-1), "phi mat should have size NxN")

    int N = delta_list.size(-1);
    auto phi_mat_cos = torch::cos(phi_mat); // left-upper triangular matrix
    auto phi_mat_sin = torch::sin(phi_mat);
    auto W = torch::ones_like(delta_list).diag_embed().contiguous();
    int BS = delta_list.numel() / N;
    phi_mat = phi_mat.contiguous();
    AT_DISPATCH_FLOATING_TYPES(W.scalar_type(), "unitaryReconstructReckCUDALauncher", ([&] {
        unitaryReconstructReckCUDALauncher<scalar_t>(
            BS,
            N,
            W.data_ptr<scalar_t>(),
            phi_mat_cos.data_ptr<scalar_t>(),
            phi_mat_sin.data_ptr<scalar_t>());
    }));

    return W.mul_(delta_list.unsqueeze(-1));


}



// void decompose_reck_complex_forward(
//   at::Tensor U,
//   at::Tensor delta_list,
//   at::Tensor phi_mat
// )
// {
//     // TORCH_CHECK(U.type().is_cuda(), "U must be a CUDA tensor")
//     TORCH_CHECK(U.size(-2) == U.size(-1), "U must be a square tensor")
//     TORCH_CHECK(U.is_contiguous(), "U must be contiguous()")
//     TORCH_CHECK(delta_list.size(-1) == U.size(-1), "delta_list should have length N")
//     TORCH_CHECK(phi_mat.size(-2) == phi_mat.size(-1) && phi_mat.size(-1) == U.size(-1), "phi mat should have size NxN")


//     auto N = U.size(-1);
//     U = U.contiguous();
//     phi_mat = phi_mat.contiguous(); // reck-style upper triangular array
//     delta_list = delta_list.contiguous();
//     int BS = U.numel() / (N * N);
//     AT_DISPATCH_FLOATING_TYPES(U.scalar_type(), "unitaryDecomposeReckCUDALauncher", ([&] {
//         unitaryDecomposeReckCUDALauncher<scalar_t>(
//             BS,
//             N,
//             U.data_ptr<scalar_t>(),
//             delta_list.data_ptr<scalar_t>(),
//             phi_mat.data_ptr<scalar_t>());
//     }));

// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decompose_francis", &decompose_francis_forward, "matrix parametrization in Francis style");
    m.def("reconstruct_francis", &reconstruct_francis_forward, "matrix parametrization reconstruct in Francis style");
    m.def("decompose_clements", &decompose_clements_forward, "matrix parametrization in Clements style");
    m.def("reconstruct_clements", &reconstruct_clements_forward, "matrix parametrization reconstruct in Clements style");
    m.def("decompose_reck", &decompose_reck_forward, "matrix parametrization in Reck style");
    m.def("reconstruct_reck", &reconstruct_reck_forward, "matrix parametrization reconstruct in Reck style");
}
