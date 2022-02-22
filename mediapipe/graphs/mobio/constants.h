#ifndef MEDIAPIPE_GRPAHS_MOBIO_CONSTANTS_H_
#define MEDIAPIPE_GRPAHS_MOBIO_CONSTANTS_H_

#include <memory>
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mobio {

// typedef struct Point {
//     uint32_t x;
//     uint32_t y;
// } Point;

typedef cv::Point Point;
typedef std::vector<Point> Points;
typedef std::vector<mediapipe::Landmark> Contour;
typedef cv::Vec3d Intensity;
enum Color {
    kColorRed = 0,
    kColorGreen = 1,
    kColorBlue = 2,
};

// Indices used for tracking landmark points in tri-region contours.
// These points were empirically selected to maximize coverage of each region
// while minimizing capture of non-smooth, non-skin artifacts 
// (eyebrows, facial hair, hairline, nose bridge, etc.)
constexpr std::array kTrackerIdxsIntensityForehead {103, 67,109, 10, 338, 297, 332, 334, 296, 336, 107, 66, 105, 104, 69, 108, 151, 337, 299, 333};
constexpr std::array kTrackerIdxsIntensityRightCheek {116, 111, 117, 118, 101, 203, 206, 216, 207, 205, 187, 147, 123, 50};
constexpr std::array kTrackerIdxsIntensityLeftCheek {345, 340, 346, 347, 330, 266, 423, 426, 436, 427, 411, 376, 352, 280, 425};

// Indices used for tracking the radius of tri-region areas.
constexpr std::array kTrackerIdxsRadiusForehead {104, 69, 108, 151, 337, 299, 333};
constexpr std::array kTrackerIdxsRadiusRightCheek {255, 261, 340, 352, 411, 427, 436};
constexpr std::array kTrackerIdxsRadiusLeftCheek {25, 31, 111, 123, 187, 207, 216};

// TODO(snair)
constexpr int kFaceLandmarkIdxExtremeLeft = 234;
constexpr int kFaceLandmarkIdxExtremeRight = 454;
}

#endif // MEDIAPIPE_GRPAHS_MOBIO_CONSTANTS_H_
