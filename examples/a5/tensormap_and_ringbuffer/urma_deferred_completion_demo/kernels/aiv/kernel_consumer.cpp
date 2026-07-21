#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);

    __gm__ float *src = reinterpret_cast<__gm__ float *>(src_tensor->buffer.addr) + src_tensor->start_offset;
    __gm__ float *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;

    constexpr int kTotalRows = 128;
    constexpr int kRows = 64;
    constexpr int kCols = 128;
    constexpr int kIters = kTotalRows / kRows;
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStrideDim5 = pto::Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    TileData src_tile(kRows, kCols);
    TileData result_tile(kRows, kCols);
    TASSIGN(src_tile, 0x0);
    TASSIGN(result_tile, 0x10000);

    constexpr int kChunkElems = kRows * kCols;
    for (int iter = 0; iter < kIters; ++iter) {
        GlobalData src_global(src + iter * kChunkElems);
        GlobalData result_global(result + iter * kChunkElems);
        TLOAD(src_tile, src_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TADDS(result_tile, src_tile, 1.0f);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(result_global, result_tile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}
