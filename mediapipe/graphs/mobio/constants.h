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

// Indices used for tracking the radius of tri-region areas.
constexpr std::array kTrackerIdxsRadiusForehead {104, 69, 108, 151, 337, 299, 333};
constexpr std::array kTrackerIdxsRadiusRightCheek {255, 261, 340, 352, 411, 427, 436};
constexpr std::array kTrackerIdxsRadiusLeftCheek {25, 31, 111, 123, 187, 207, 216};

// TODO(snair)
constexpr int kFaceLandmarkIdxExtremeLeft = 234;
constexpr int kFaceLandmarkIdxExtremeRight = 454;
}

#endif // MEDIAPIPE_GRPAHS_MOBIO_CONSTANTS_H_
