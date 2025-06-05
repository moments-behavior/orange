#include "shaman.h"
#include "common.hpp"

inline std::vector<shaman::Object> conv_shaman(std::vector<pose::Object>& objs) {

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
        std::copy(p.kps.begin(), p.kps.end(), o.kps);
        
        shaman_objs.push_back(std::move(o));
    }
    return shaman_objs;

}

