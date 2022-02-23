licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_library(
    name = "mobio_facial_analysis_main",
    srcs = ["mobio_facial_analysis_main.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/graphs/mobio:constants",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_binary(
    name = "mobio_facial_analysis",
    deps = [
        "//mediapipe/examples/desktop/mobio:mobio_facial_analysis_main",
        "//mediapipe/graphs/mobio:desktop_live_calculators",
    ],
)
