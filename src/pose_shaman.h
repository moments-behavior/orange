#include "shaman.h"
#include "common.hpp"
#include <vector>
#include <algorithm> // For std::copy

inline std::vector<shaman::Object> conv_shaman(const std::vector<pose::Object>& objs) {

    std::vector<shaman::Object> shaman_objs;
    shaman_objs.reserve(objs.size());

    for (const auto& p : objs) {
        shaman::Object o;
        o.rect.x = p.rect.x;
        o.rect.y = p.rect.y;
        o.rect.width = p.rect.width;
        o.rect.height = p.rect.height;
        o.label = p.label;
        o.prob = p.prob;

        // Copy from the C-style array using the num_kps member
        if (p.num_kps > 0) {
            std::copy(p.kps, p.kps + p.num_kps, o.kps);
        }
        o.num_kps = p.num_kps;
        
        shaman_objs.push_back(std::move(o));
    }
    return shaman_objs;
}