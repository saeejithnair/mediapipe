# Author: Saeejith Nair (smnair@uwaterloo.ca)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobio_contour_calculator",
    srcs = ["mobio_contour_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/graphs/mobio:constants",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mobio_intensity_calculator",
    srcs = ["mobio_intensity_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/graphs/mobio:constants",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
    ],
    alwayslink = 1,
)