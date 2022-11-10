#pragma once
#include <memory>
#include <mutex>
#include <vector>
#include <map>

namespace vins_slam {
using namespace std;

class FeatureMeta {
public:
    int feature_id;
    // store the left and right coordinate
    double u0;
    double v0;
    double u1;
    double v1;
    double un_u0;
    double un_v0;
    double un_u1;
    double un_v1;
    float response;
    int life_time;
};

typedef struct FeatureInfo {
    std::vector<FeatureMeta> frame_feature;
    double frame_timestamp;
} FeatureInfo_t;

typedef std::map<int, std::vector<FeatureMeta>> GridFeature;

}