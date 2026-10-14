// Host-side driver for SignalQuality: reads raw int16 LE @ 8 kHz from a file,
// feeds it in 800-sample chunks (like the UART frames) and prints one CSV row per result.
#include "signal_quality.hpp"
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s pcm16_8k.raw\n", argv[0]); return 1; }
    FILE* fp = std::fopen(argv[1], "rb");
    if (!fp) return 1;
    std::vector<int16_t> pcm;
    int16_t buf[800];
    size_t n;
    while ((n = std::fread(buf, 2, 800, fp)) > 0) pcm.insert(pcm.end(), buf, buf + n);
    std::fclose(fp);

    SignalQuality sq;
    QualityTracker tr;
    std::vector<SignalQuality::Result> res;
    for (size_t i = 0; i < pcm.size(); i += 800)
        sq.push(pcm.data() + i, std::min<size_t>(800, pcm.size() - i), res);
    for (const auto& r : res) {
        tr.add(r);
        std::printf("%.3f,%.3f,%.17g,%s,%s,%.17g,%.17g,%.17g", r.t_start_s, r.t_end_s, r.q,
                    SignalQuality::verdictName(r.verdict), SignalQuality::reasonName(r.reason),
                    r.level_dbfs, r.clip_frac, r.hr_bpm);
        for (double f : r.features) std::printf(",%.17g", f);
        std::printf("\n");
    }
    const auto& s = tr.summary();
    std::printf("#summary,%.1f,%.1f,%.1f,%.1f,%.4f,%.3f,%.1f,%.4f\n", s.scored_s, s.good_s, s.marginal_s,
                s.bad_s, s.usable_ratio, s.usable_beats, s.longest_run_s, s.mean_q);
    return 0;
}
