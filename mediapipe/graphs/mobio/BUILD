# Author: Saeejith Nair (smnair@uwaterloo.ca)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "desktop_live_calculators",
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/graphs/face_mesh/subgraphs:face_renderer_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/graphs/mobio/subgraphs:mobio_result_processor",
    ],
)
