#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "scalar_kernels.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace spacemit_kernels::scalar {

// ── L2 NORM ──────────────────────────────────────────────────────────────────
void forward_l2_norm_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));
    if (eps < 0.0f) eps = 0.0f;

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb0  = op->nb[0],   nb1  = op->nb[1],   nb2  = op->nb[2],   nb3  = op->nb[3];

    const int64_t nr    = ne01 * ne02 * ne03;
    const int64_t dr    = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0   = dr * ctx.ith;
    const int64_t ir1   = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        const float * x = (const float *)((const char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float *       y = (float *)((char *)op->data + i01*nb1 + i02*nb2 + i03*nb3);

        double sum = 0.0;
        for (int64_t i = 0; i < ne00; ++i) sum += (double)(x[i] * x[i]);
        const float scale = 1.0f / fmaxf(sqrtf((float)sum), eps);
        for (int64_t i = 0; i < ne00; ++i) y[i] = x[i] * scale;
    }
}

// ── FILL ─────────────────────────────────────────────────────────────────────
void forward_fill_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const float c = ggml_get_op_params_f32(op, 0);

    const int64_t ne0 = op->ne[0], ne1 = op->ne[1], ne2 = op->ne[2], ne3 = op->ne[3];
    const size_t  nb1 = op->nb[1], nb2 = op->nb[2], nb3 = op->nb[3];

    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
        float * dst_row = (float *)((char *)op->data + i3*nb3 + i2*nb2 + i1*nb1);
        for (int64_t i = 0; i < ne0; ++i) dst_row[i] = c;
    }
}

// ── CUMSUM ────────────────────────────────────────────────────────────────────
void forward_cumsum_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb1  = op->nb[1],   nb2  = op->nb[2],   nb3  = op->nb[3];

    const int64_t nr  = ne01 * ne02 * ne03;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03*ne02*ne01) / ne01;
        const int64_t i01 = ir - i03*ne02*ne01 - i02*ne01;

        const float * src_row = (const float *)((const char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float *       dst_row = (float *)((char *)op->data + i01*nb1 + i02*nb2 + i03*nb3);

        float acc = 0.0f;
        for (int64_t i = 0; i < ne00; ++i) { acc += src_row[i]; dst_row[i] = acc; }
    }
}

// ── PAD ───────────────────────────────────────────────────────────────────────
void forward_pad_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int64_t ne0 = op->ne[0], ne1 = op->ne[1], ne2 = op->ne[2], ne3 = op->ne[3];
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb1  = op->nb[1],   nb2  = op->nb[2],   nb3  = op->nb[3];

    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3*ne2*ne1) / ne1;
        const int64_t i1 = ir - i3*ne2*ne1 - i2*ne1;

        float * dst_row = (float *)((char *)op->data + i3*nb3 + i2*nb2 + i1*nb1);

        if (i3 < ne03 && i2 < ne02 && i1 < ne01) {
            const float * src_row = (const float *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
            const int64_t copy_n = std::min(ne0, ne00);
            for (int64_t i = 0; i < copy_n; ++i) dst_row[i] = src_row[i];
            for (int64_t i = copy_n; i < ne0; ++i) dst_row[i] = 0.0f;
        } else {
            for (int64_t i = 0; i < ne0; ++i) dst_row[i] = 0.0f;
        }
    }
}

// ── TRI ───────────────────────────────────────────────────────────────────────
void forward_tri_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(op, 0);

    const int64_t ne0 = src0->ne[0], ne1 = src0->ne[1], ne2 = src0->ne[2], ne3 = src0->ne[3];
    const int64_t nr  = ne1 * ne2 * ne3;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3*ne2*ne1) / ne1;
        const int64_t i1 = ir - i3*ne2*ne1 - i2*ne1;  // row index

        const float * src_row = (const float *)src0->data + (i3*ne2*ne1 + i2*ne1 + i1) * ne0;
        float *       dst_row = (float *)op->data          + (i3*ne2*ne1 + i2*ne1 + i1) * ne0;

        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            bool keep;
            switch (ttype) {
                case GGML_TRI_TYPE_LOWER:      keep = i0 <  i1; break;
                case GGML_TRI_TYPE_LOWER_DIAG: keep = i0 <= i1; break;
                case GGML_TRI_TYPE_UPPER:      keep = i0 >  i1; break;
                case GGML_TRI_TYPE_UPPER_DIAG: keep = i0 >= i1; break;
                default: keep = false;
            }
            dst_row[i0] = keep ? src_row[i0] : 0.0f;
        }
    }
}

// ── DIAG ──────────────────────────────────────────────────────────────────────
void forward_diag_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    // src0 is 1D vector [n], dst is 2D [n,n]
    const int64_t n   = src0->ne[0];
    const int64_t dr  = (n + ctx.nth - 1) / ctx.nth;
    const int64_t i0  = dr * ctx.ith, i1 = std::min(i0 + dr, n);

    // zero the output first (only thread 0 to avoid races)
    if (ctx.ith == 0) {
        memset(op->data, 0, ggml_nbytes(op));
    }
    ctx.sync();

    const float * src = (const float *)src0->data;
    for (int64_t i = i0; i < i1; ++i) {
        float * dst_row = (float *)((char *)op->data + i * op->nb[1]);
        dst_row[i] = src[i];
    }
}

// ── SET ───────────────────────────────────────────────────────────────────────
void forward_set_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];  // destination tensor data (will be copied to output)
    const ggml_tensor * src1 = op->src[1];  // source of values to set

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    // op params: nb1, nb2, nb3, offset
    const size_t  nb1    = ((const int32_t *)op->op_params)[0];
    const size_t  nb2    = ((const int32_t *)op->op_params)[1];
    const size_t  nb3    = ((const int32_t *)op->op_params)[2];
    const size_t  offset = ((const int32_t *)op->op_params)[3];

    // copy src0 to dst (thread 0 only for simplicity)
    if (ctx.ith == 0) {
        if (op->data != src0->data) {
            memcpy(op->data, src0->data, ggml_nbytes(src0));
        }
    }
    ctx.sync();

    // now overlay src1 at the offset
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const size_t  nb10 = src1->nb[0];

    const int64_t nr  = ne11 * ne12 * ne13;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith, ir1 = std::min(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i3 = ir / (ne12 * ne11);
        const int64_t i2 = (ir - i3*ne12*ne11) / ne11;
        const int64_t i1 = ir - i3*ne12*ne11 - i2*ne11;

        const float * src_row = (const float *)((const char *)src1->data + i1*src1->nb[1] + i2*src1->nb[2] + i3*src1->nb[3]);
        float *       dst_row = (float *)((char *)op->data + offset + i1*nb1 + i2*nb2 + i3*nb3);

        for (int64_t i = 0; i < ne10; ++i) {
            dst_row[i] = src_row[i];
        }
    }
}

// ── MUL_MAT F32 × F32 ────────────────────────────────────────────────────────
// dst[m,n] = sum_k(src0[k,m] * src1[k,n])
void forward_mul_mat_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int64_t M   = src0->ne[1];
    const int64_t K   = src0->ne[0];
    const int64_t N   = src1->ne[1];
    const int64_t B2  = src0->ne[2];  // batch dim 2
    const int64_t B3  = src0->ne[3];  // batch dim 3

    const int64_t total_rows = M * B2 * B3;
    const int64_t dr = (total_rows + ctx.nth - 1) / ctx.nth;
    const int64_t r0 = dr * ctx.ith, r1 = std::min(r0 + dr, total_rows);

    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i3 = r / (B2 * M);
        const int64_t i2 = (r - i3 * B2 * M) / M;
        const int64_t m  = r - i3 * B2 * M - i2 * M;

        const float * w_row = (const float *)((const char *)src0->data + m * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);

        // src1 batch: broadcast or direct
        const int64_t s1_i2 = i2 % src1->ne[2];
        const int64_t s1_i3 = i3 % src1->ne[3];

        float * d_row = (float *)((char *)op->data + m * op->nb[1] + i2 * op->nb[2] + i3 * op->nb[3]);

        for (int64_t n = 0; n < N; ++n) {
            const float * x_row = (const float *)((const char *)src1->data + n * src1->nb[1] + s1_i2 * src1->nb[2] + s1_i3 * src1->nb[3]);
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += w_row[k] * x_row[k];
            d_row[n] = acc;
        }
    }
}

// ── SOLVE_TRI ─────────────────────────────────────────────────────────────────
// Forward/backward triangular solve — scalar fallback
void forward_solve_tri_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    // Only thread 0 executes; others skip and wait
    if (ctx.ith != 0) return;

    const ggml_tensor * src0 = op->src[0];  // triangular matrix A
    const ggml_tensor * src1 = op->src[1];  // right-hand side B

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);

    const int upper    = ggml_get_op_params_i32(op, 0);  // 1=upper, 0=lower
    const int transpose_A = ggml_get_op_params_i32(op, 1);

    const int64_t n = src0->ne[0];  // matrix size
    const int64_t nrhs = src1->ne[1];

    // copy src1 → dst first
    memcpy(op->data, src1->data, ggml_nbytes(src1));

    float * X = (float *)op->data;
    const float * A = (const float *)src0->data;

    // Simple triangular solve (forward substitution for lower, back for upper)
    for (int64_t j = 0; j < nrhs; ++j) {
        if (!upper && !transpose_A) {
            // lower triangular, no transpose
            for (int64_t i = 0; i < n; ++i) {
                float s = X[j * n + i];
                for (int64_t k = 0; k < i; ++k) s -= A[i * n + k] * X[j * n + k];
                X[j * n + i] = s / A[i * n + i];
            }
        } else {
            // upper triangular, no transpose (back substitution)
            for (int64_t i = n - 1; i >= 0; --i) {
                float s = X[j * n + i];
                for (int64_t k = i + 1; k < n; ++k) s -= A[i * n + k] * X[j * n + k];
                X[j * n + i] = s / A[i * n + i];
            }
        }
    }
}

// ── SSM CONV ──────────────────────────────────────────────────────────────────
// Ported from ggml_compute_forward_ssm_conv_f32, parallelised over d_inner rows.
void forward_ssm_conv_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0]; // conv_x  {d_conv-1+n_t, d_inner, n_seqs}
    const ggml_tensor * src1 = op->src[1]; // weight  {d_conv, d_inner}

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const int nc  = (int)src1->ne[0]; // d_conv
    const int ncs = (int)src0->ne[0]; // d_conv - 1 + n_t
    const int nr  = (int)src0->ne[1]; // d_inner
    const int n_t = (int)op->ne[1];   // tokens per sequence
    const int n_s = (int)op->ne[2];   // number of sequences

    GGML_ASSERT(op->ne[0] == nr);

    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = std::min((int64_t)nr, ir0 + dr);

    for (int i3 = 0; i3 < n_s; ++i3) {
        for (int i2 = 0; i2 < n_t; ++i2) {
            const float * s = (const float *)((const char *)src0->data
                + ir0 * src0->nb[1] + i2 * src0->nb[0] + i3 * src0->nb[2]);
            const float * c = (const float *)((const char *)src1->data
                + ir0 * src1->nb[1]);
            float * x = (float *)((char *)op->data
                + ir0 * op->nb[0] + i2 * op->nb[1] + i3 * op->nb[2]);

            const int ir = (int)(ir1 - ir0);
            for (int i1 = 0; i1 < ir; ++i1) {
                float sumf = 0.0f;
                for (int i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1 * ncs] * c[i0 + i1 * nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

// ── GATED DELTA NET ───────────────────────────────────────────────────────────
// Ported from ggml_compute_forward_gated_delta_net_one_chunk.
// Parallelised over heads × sequences (ir = head_index + seq * H).
void forward_gated_delta_net(ggml::spacemit::context & ctx, ggml_tensor * op) {
    ggml_tensor * src_q     = op->src[0];
    ggml_tensor * src_k     = op->src[1];
    ggml_tensor * src_v     = op->src[2];
    ggml_tensor * src_g     = op->src[3];
    ggml_tensor * src_beta  = op->src[4];
    ggml_tensor * src_state = op->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    const int64_t K = ggml_get_op_params_i32(op, 0);
    GGML_ASSERT(K >= 1);

    const int64_t nr  = H * n_seqs;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = std::min(nr, ir0 + dr);

    const int64_t state_seq_stride = src_state->nb[3] / sizeof(float);
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;

    float * attn_out_base  = (float *)op->data;
    float * state_out_base = (float *)op->data + attn_score_elems;
    const float * state_in_base = (const float *)src_state->data;

    const float scale = 1.0f / sqrtf((float)S_v);
    const bool kda = (src_g->ne[0] == S_v);

    // scratch buffer for delta (S_v floats) and optional state_work (S_v*S_v floats)
    std::vector<float> scratch((size_t)(S_v + (K > 1 ? S_v * S_v : 0)));
    float * delta      = scratch.data();
    float * state_work = K > 1 ? (delta + S_v) : nullptr;

    // local tensor nb helpers
    const size_t nbq1 = src_q->nb[1], nbq2 = src_q->nb[2], nbq3 = src_q->nb[3];
    const size_t nbk1 = src_k->nb[1], nbk2 = src_k->nb[2], nbk3 = src_k->nb[3];
    const size_t nbv1 = src_v->nb[1], nbv2 = src_v->nb[2], nbv3 = src_v->nb[3];
    const size_t nbg1 = src_g->nb[1], nbg2 = src_g->nb[2], nbg3 = src_g->nb[3];
    const size_t nbb1 = src_beta->nb[1], nbb2 = src_beta->nb[2], nbb3 = src_beta->nb[3];
    const int64_t neq1 = src_q->ne[1], neq3 = src_q->ne[3];
    const int64_t nek1 = src_k->ne[1], nek3 = src_k->ne[3];
    const int64_t nev3 = src_v->ne[3];
    const int64_t rq3  = nev3 / neq3;
    const int64_t rk3  = nev3 / nek3;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H;   // head index
        const int64_t iv3 = ir / H;   // sequence index

        const int64_t iq1 = iv1 % neq1;
        const int64_t ik1 = iv1 % nek1;
        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        float * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        const float * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, (size_t)(S_v * S_v) * sizeof(float));

        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        for (int64_t t = 0; t < n_tokens; t++) {
            const float * q_d = (const float *)((const char *)src_q->data + iq3*nbq3 + t*nbq2 + iq1*nbq1);
            const float * k_d = (const float *)((const char *)src_k->data + ik3*nbk3 + t*nbk2 + ik1*nbk1);
            const float * v_d = (const float *)((const char *)src_v->data + iv3*nbv3 + t*nbv2 + iv1*nbv1);
            const float   beta_val = *(const float *)((const char *)src_beta->data + iv3*nbb3 + t*nbb2 + iv1*nbb1);
            const float * g_d      =  (const float *)((const char *)src_g->data    + iv3*nbg3 + t*nbg2 + iv1*nbg1);

            if (kda) {
                for (int64_t i = 0; i < S_v; ++i) delta[i] = expf(g_d[i]);
                for (int64_t j = 0; j < S_v; ++j) {
                    float * row = &s_out[j * S_v];
                    for (int64_t i = 0; i < S_v; ++i) row[i] *= delta[i];
                }
            } else {
                float eg = expf(g_d[0]);
                for (int64_t i = 0; i < S_v * S_v; ++i) s_out[i] *= eg;
            }

            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                const float * row = &s_out[j * S_v];
                for (int64_t i = 0; i < S_v; ++i) sum += row[i] * k_d[i];
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            for (int64_t j = 0; j < S_v; ++j) {
                float * row = &s_out[j * S_v];
                for (int64_t i = 0; i < S_v; ++i) row[i] += k_d[i] * delta[j];
            }

            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                const float * row = &s_out[j * S_v];
                for (int64_t i = 0; i < S_v; ++i) sum += row[i] * q_d[i];
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H;

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * curr_state_o = state_out_base + target_slot * state_size_per_snap
                                         + (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, (size_t)(S_v * S_v) * sizeof(float));
                }
            }
        }
    }
}

} // namespace spacemit_kernels::scalar
