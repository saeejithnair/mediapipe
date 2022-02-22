#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

#include "mediapipe/graphs/mobio/constants.h"

namespace mediapipe {

namespace {

constexpr char kImageTag[] = "IMAGE";
constexpr char kNumFacesTag[] = "NUM_FACES";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kContourIntensityForeheadTag[] = "CONTOUR_INTENSITY_FOREHEAD";
constexpr char kContourIntensityLCheekTag[] = "CONTOUR_INTENSITY_LCHEEK";
constexpr char kContourIntensityRCheekTag[] = "CONTOUR_INTENSITY_RCHEEK";

typedef std::vector<NormalizedLandmarkList> NormalizedLandmarkLists;
} // anonymous namespace

class MobioContourCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:

  template<std::size_t N>
  absl::Status ComputeContourFromTrackers(
                const int img_width, const int img_height,
                const NormalizedLandmarkList& landmark_list, 
                const std::array<int, N>& trackers,
                mobio::Points* contour_points);

};

REGISTER_CALCULATOR(MobioContourCalculator);

absl::Status MobioContourCalculator::GetContract(CalculatorContract* cc) {
  // Validate input/output stream tags connected to node.
  RET_CHECK(cc->Inputs().HasTag(kImageTag));
  RET_CHECK(cc->Inputs().HasTag(kLandmarksTag));
  RET_CHECK(cc->Outputs().HasTag(kContourIntensityForeheadTag));
  RET_CHECK(cc->Outputs().HasTag(kContourIntensityLCheekTag));
  RET_CHECK(cc->Outputs().HasTag(kContourIntensityRCheekTag));
  
  // Set type for input stream packets.
  cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
  cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkLists>();

  // Set type for num faces side packet.
  auto& side_inputs = cc->InputSidePackets();
  side_inputs.Tag(kContourIntensityForeheadTag).Set<int>();

  // Set type for output stream packets.
  cc->Outputs().Tag(kContourIntensityForeheadTag).Set<mobio::Points>();
  cc->Outputs().Tag(kContourIntensityLCheekTag).Set<mobio::Points>();
  cc->Outputs().Tag(kContourIntensityRCheekTag).Set<mobio::Points>();

  return absl::OkStatus();
}

absl::Status MobioContourCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));


  return absl::OkStatus();
}

template<std::size_t N>
absl::Status MobioContourCalculator::ComputeContourFromTrackers(
    const int img_width, const int img_height,
    const NormalizedLandmarkList& landmark_list, 
    const std::array<int, N>& trackers,
    mobio::Points* contour_points) {
    
    RET_CHECK(contour_points != NULL);

    // Validate that contour_points is empty (size equals 0). 
    // We expect 1:1 mapping from tracker index to tracker point.
    RET_CHECK(!contour_points->size());
    
    // Allocate space to hold points for all trackers.
    contour_points->reserve(N);

    for (int i=0; i<N; ++i) {
      const auto& landmark = landmark_list.landmark(trackers[i]);
      contour_points->emplace_back(cv::Point2d(
                                    landmark.x()*img_width, 
                                    landmark.y()*img_height));
    }

    return absl::OkStatus();
}

absl::Status MobioContourCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kImageTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const auto& input_frame = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
  const auto input_frame_width = input_frame.Width();
  const auto input_frame_height = input_frame.Height();

  // Get vector of landmark list.
  const auto& landmark_lists = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkLists>();

  // Get face landmarks. Raise error if more than 1 face is detected. 
  const auto num_faces = cc->InputSidePackets().Tag(kNumFacesTag).Get<int>();
  RET_CHECK_EQ(num_faces, 1) << "Expected 1 face, received " 
                             << num_faces << " faces.";
  const auto& landmark_list = landmark_lists.front();

  // Create vectors for storing tracker points for each face region of interest.
  mobio::Points contour_points_intensity_forehead;
  mobio::Points contour_points_intensity_lcheek;
  mobio::Points contour_points_intensity_rcheek;

  // Compute tracker points for each face region of interest.
  RET_CHECK_OK(ComputeContourFromTrackers(
                input_frame_width, input_frame_height,
                landmark_list, 
                mobio::kTrackerIdxsIntensityForehead, 
                &contour_points_intensity_forehead));

  RET_CHECK_OK(ComputeContourFromTrackers(
                input_frame_width, input_frame_height,
                landmark_list, 
                mobio::kTrackerIdxsIntensityLeftCheek, 
                &contour_points_intensity_lcheek));

  RET_CHECK_OK(ComputeContourFromTrackers(
                input_frame_width, input_frame_height,
                landmark_list, 
                mobio::kTrackerIdxsIntensityRightCheek, 
                &contour_points_intensity_rcheek));

  // Publish outputs.
  cc->Outputs().Tag(kContourIntensityForeheadTag).Add(&contour_points_intensity_forehead, cc->InputTimestamp());
  cc->Outputs().Tag(kContourIntensityLCheekTag).Add(&contour_points_intensity_lcheek, cc->InputTimestamp());
  cc->Outputs().Tag(kContourIntensityRCheekTag).Add(&contour_points_intensity_rcheek, cc->InputTimestamp());

  return absl::OkStatus();
}

} // namespace mediapipe
