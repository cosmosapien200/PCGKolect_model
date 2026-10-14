// Host driver for SegQuality parity tests.
//   seg_cli feat   window.f32 out.f32        features of one kSamples window
//   seg_cli decode P.f32 T                   viterbi + beats on posteriors [T][5]
//   seg_cli stream pcm16_8k.raw [replay.f32] full streaming path; the NPU is replaced by either a
//                                            deterministic stub or a replay of per-window posteriors
#include "seg_quality.hpp"
#include "seg_params.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template <class T> static std::vector<T> readAll(const char* path) {
    std::vector<T> v;
    FILE* fp = std::fopen(path, "rb");
    if (!fp) { std::perror(path); std::exit(1); }
    T buf[4096];
    size_t n;
    while ((n = std::fread(buf, sizeof(T), 4096, fp)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(fp);
    return v;
}

static void printBeat(const SegQuality::Beat& b) {
    std::printf("beat %ld %ld %.17g %.17g %d\n", b.start_frame, b.end_frame, b.conf, b.score, b.usable ? 1 : 0);
}

// Stub "NPU": logits[t][c] = 3 * feat[band 6c][t]  (mirrored in test_seg_parity.py)
static bool stubInfer(const float* feat, float* probs) {
    using namespace seg_params;
    for (int t = 0; t < kFrames; t++) {
        double l[5], m = -1e300, s = 0;
        for (int c = 0; c < 5; c++) { l[c] = 3.0 * feat[(6 * c) * kFrames + t]; m = std::max(m, l[c]); }
        for (int c = 0; c < 5; c++) { l[c] = std::exp(l[c] - m); s += l[c]; }
        for (int c = 0; c < 5; c++) probs[t * 5 + c] = static_cast<float>(l[c] / s);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    if (!std::strcmp(argv[1], "feat")) {
        auto x = readAll<float>(argv[2]);
        std::vector<float> f(seg_params::kBands * seg_params::kFrames);
        SegQuality::features(x.data(), f.data());
        FILE* fp = std::fopen(argv[3], "wb");
        std::fwrite(f.data(), sizeof(float), f.size(), fp);
        std::fclose(fp);
    } else if (!std::strcmp(argv[1], "decode")) {
        auto P = readAll<float>(argv[2]);
        const int T = std::atoi(argv[3]);
        std::vector<int> cls;
        SegQuality::viterbi(P.data(), T, cls);
        std::printf("cls");
        for (int c : cls) std::printf(" %d", c);
        std::printf("\n");
        std::vector<SegQuality::Beat> bs;
        SegQuality::beats(cls, P.data(), T, bs);
        for (auto& b : bs) printBeat(b);
    } else if (!std::strcmp(argv[1], "stream")) {
        auto pcm = readAll<int16_t>(argv[2]);
        std::vector<float> replay;
        size_t k = 0;
        if (argc > 3) replay = readAll<float>(argv[3]);
        const size_t per = seg_params::kFrames * 5;
        SegQuality sq(replay.empty() ? SegQuality::InferFn(stubInfer) : SegQuality::InferFn(
            [&](const float*, float* probs) {
                if ((k + 1) * per > replay.size()) return false;
                std::copy(replay.begin() + k * per, replay.begin() + (k + 1) * per, probs);
                k++;
                return true;
            }));
        std::vector<SegQuality::Beat> bs;
        for (size_t i = 0; i < pcm.size(); i += 800) sq.push(pcm.data() + i, std::min<size_t>(800, pcm.size() - i), bs);
        for (auto& b : bs) printBeat(b);
    }
    return 0;
}
